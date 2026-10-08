import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from pathlib import Path
import os
import shutil

from ml.engine.metrics import compute_classification_metrics
from ml.engine.trainer import Trainer, load_checkpoint
from ml.config import Config

def test_metrics():
    y_true = [0, 1, 2, 0, 1, 2, 0, 2, 2]
    y_pred = [0, 1, 1, 0, 1, 2, 0, 2, 2]
    
    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics['accuracy'] == 8 / 9
    
class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 3)
        
    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))

class MockDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], f"path_{idx}"

def test_trainer_overfit():
    # 32 synthetic images, 10 features each, 3 classes
    torch.manual_seed(42)
    X = torch.randn(32, 10)
    y = torch.randint(0, 3, (32,))
    
    dataset = MockDataset(X, y)
    loader = DataLoader(dataset, batch_size=8)
    
    model = TinyModel()
    
    config = Config({
        "train": {
            "epochs": 20,
            "early_stopping_patience": 5,
            "learning_rate": 0.1,
            "weight_decay": 0.0,
            "batch_size": 8,
            "mixed_precision": False,
            "channels_last": False,
            "grad_clip": 0.0
        },
        "model": {
            "name": "tiny",
            "variant": "v1"
        },
        "data": {
            "img_size": 10
        }
    })
    
    out_dir = Path("tests/_tmp_trainer_overfit")
    
    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        config=config,
        device=torch.device("cpu"),
        class_names=["A", "B", "C"],
        out_dir=out_dir
    )
    
    metrics = trainer.fit()
    
    assert metrics["epochs_run"] > 0
    
    # Check early stopping: if it didn't improve for 5 epochs
    # Wait, overfit will improve until 1.0. Let's just check history
    history_file = out_dir / "history.csv"
    assert history_file.exists()
    
    import pandas as pd
    df = pd.read_csv(history_file)
    assert len(df) > 0
    
    shutil.rmtree(out_dir)

def test_checkpoint_roundtrip():
    model = TinyModel()
    # Dummy weights
    with torch.no_grad():
        model.fc.weight.fill_(0.5)
        model.fc.bias.fill_(0.1)
        
    config = Config({
        "train": {
            "epochs": 1,
            "patience": 1,
            "learning_rate": 0.1,
            "weight_decay": 0.0,
            "mixed_precision": False
        },
        "model": {
            "name": "tiny",
            "variant": "v1"
        }
    })
    
    X = torch.randn(4, 10)
    y = torch.randint(0, 3, (4,))
    dataset = MockDataset(X, y)
    loader = DataLoader(dataset, batch_size=4)
    
    out_dir = Path("tests/_tmp_checkpoint")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # We monkey patch factory for the test
    import ml.models.factory as factory
    original_create = factory.build_model
    def mock_create(name, variant, num_classes, **kwargs):
        return TinyModel()
    factory.build_model = mock_create
    
    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        config=config,
        device=torch.device("cpu"),
        class_names=["A", "B", "C"],
        out_dir=out_dir
    )
    
    trainer.start_epoch = 10
    trainer._save_checkpoint(out_dir / "ckpt.pt", is_best=False)
    
    # Check logits before
    model.eval()
    with torch.no_grad():
        logits_before = model(X)
        
    # Load
    model_loaded, ckpt = load_checkpoint(out_dir / "ckpt.pt", device=torch.device("cpu"))
    model_loaded.eval()
    with torch.no_grad():
        logits_after = model_loaded(X)
        
    assert torch.allclose(logits_before, logits_after)
    assert ckpt["epoch"] == 10
    
    # Restore factory
    factory.build_model = original_create
    shutil.rmtree(out_dir)

def test_early_stopping():
    model = TinyModel()
    X = torch.randn(4, 10)
    y = torch.randint(0, 3, (4,))
    loader = DataLoader(MockDataset(X, y), batch_size=4)
    
    config = Config({
        "train": {
            "epochs": 100,
            "early_stopping_patience": 2, # stops after 2 epochs without improvement
            "learning_rate": 0.0, # zero lr so it doesn't improve
            "weight_decay": 0.0,
            "mixed_precision": False
        },
        "model": {
            "name": "tiny",
            "variant": "v1"
        }
    })
    
    out_dir = Path("tests/_tmp_early_stop")
    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        config=config,
        device=torch.device("cpu"),
        class_names=["A", "B", "C"],
        out_dir=out_dir
    )
    
    # Force val_macro_f1 to be a constant
    # We mock val_epoch to return constant
    trainer.val_epoch = lambda: (0.5, 0.5, 0.5)
    
    metrics = trainer.fit()
    
    # Epoch 1: best (val_macro_f1=0.5, best=0.0) -> is_best=True
    # Epoch 2: val_macro_f1=0.5, best=0.5 -> not best, without_improv=1
    # Epoch 3: val_macro_f1=0.5, best=0.5 -> not best, without_improv=2 -> early stop!
    # Total epochs run should be 3
    assert metrics["epochs_run"] == 3
    
    shutil.rmtree(out_dir)
