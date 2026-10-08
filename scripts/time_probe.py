import argparse
import time
import torch
from pathlib import Path

from ml.config import load_config
from ml.data.loaders import build_loaders
from ml.models.factory import build_model

def main():
    parser = argparse.ArgumentParser(description="Time probe to estimate training time.")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML.")
    args = parser.parse_args()

    config = load_config(args.config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Time probe using device: {device}")
    
    train_loader, val_loader, test_loader, data_info = build_loaders(config, smoke=False)
    
    model_kwargs = config.get("model", {}).copy()
    model_kwargs.pop("name", None)
    model_kwargs.pop("variant", None)
    
    model = build_model(
        name=config.model.name,
        variant=config.model.variant,
        num_classes=data_info.K,
        **model_kwargs
    )
    model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))
    
    print("Running 20 training batches...")
    model.train()
    
    start_train = time.time()
    
    for i, (images, targets, _) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if i >= 19:
            break
            
    train_time = time.time() - start_train
    time_per_train_batch = train_time / 20.0
    
    print("Running 10 validation batches...")
    model.eval()
    start_val = time.time()
    with torch.no_grad():
        for i, (images, targets, _) in enumerate(val_loader):
            images = images.to(device)
            targets = targets.to(device)
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, targets)
            if i >= 9:
                break
                
    val_time = time.time() - start_val
    time_per_val_batch = val_time / 10.0
    
    total_train_batches = len(train_loader)
    total_val_batches = len(val_loader)
    
    epoch_train_time = time_per_train_batch * total_train_batches
    epoch_val_time = time_per_val_batch * total_val_batches
    
    total_epoch_time = epoch_train_time + epoch_val_time
    
    epochs = config.train.epochs
    total_run_time = total_epoch_time * epochs
    
    print(f"\n--- TIME PROBE PROJECTION ---")
    print(f"Device: {device}")
    print(f"Time per train batch: {time_per_train_batch*1000:.1f} ms")
    print(f"Time per val batch:   {time_per_val_batch*1000:.1f} ms")
    print(f"Projected epoch time: {total_epoch_time:.1f} s (Train: {epoch_train_time:.1f}s | Val: {epoch_val_time:.1f}s)")
    print(f"Projected total run time ({epochs} epochs): {total_run_time/60:.1f} mins ({total_run_time/3600:.1f} hours)")
    print("-----------------------------\n")

if __name__ == "__main__":
    main()
