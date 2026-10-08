from pathlib import Path
from ml.config import load_config, Config

def test_config_defaults(tmp_path):
    config = load_config(overrides=["train.lr=0.5", "train.augment=true", "model.name=resnet50"])
    
    assert config.train.lr == 0.5
    assert config.train.augment is True
    assert config.model.name == "resnet50"
    assert config.project.name == "MargDrishti"
