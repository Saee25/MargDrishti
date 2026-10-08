import argparse
import sys
import time
import json
import yaml
import shutil
import platform
import subprocess
from pathlib import Path

import torch

from ml.config import load_config
from ml.data.loaders import build_loaders
from ml.models.factory import build_model, count_parameters
from ml.engine.trainer import Trainer
from ml.engine.evaluator import evaluate
from ml.utils.results_log import append_to_results_log

def get_env_info():
    info = {
        "os": platform.platform(),
        "python": sys.version,
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available()
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info

def print_start_banner(run_name, model_name, param_count, device, train_size, val_size, test_size, K, config):
    print("=" * 60)
    print(f"RUN NAME:      {run_name}")
    print(f"MODEL:         {model_name} (Variant: {config.model.variant})")
    print(f"PARAMETERS:    {param_count:,}")
    print(f"DEVICE:        {device}")
    print(f"CLASSES (K):   {K}")
    print(f"DATA SPLITS:   Train: {train_size} | Val: {val_size} | Test: {test_size}")
    print("-" * 60)
    print(f"EPOCHS:        {config.train.epochs}")
    print(f"BATCH SIZE:    {config.train.batch_size}")
    print(f"LEARNING RATE: {config.train.get('lr')}")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Train a traffic sign classification model.")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML.")
    parser.add_argument("--run-name", type=str, required=True, help="Name of the run.")
    parser.add_argument("--smoke", action="store_true", help="Run a quick smoke test.")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing run directory.")
    parser.add_argument("overrides", nargs="*", help="key=value config overrides")
    
    args = parser.parse_args()
    
    run_name = args.run_name
    if args.smoke:
        run_name = f"_smoke_{run_name}"
        
    out_dir = Path("experiments") / run_name
    if out_dir.exists() and not args.force and not args.resume:
        print(f"Error: Directory {out_dir} already exists. Use --force to overwrite or --resume to continue.")
        sys.exit(1)
        
    if out_dir.exists() and args.force and not args.resume:
        shutil.rmtree(out_dir)
        
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging to file
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "a", encoding="utf-8")
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
        def flush(self):
            self.terminal.flush()
            self.log.flush()
            
    sys.stdout = Logger(out_dir / "train_log.txt")
    
    if args.smoke:
        print("!!! SMOKE TEST !!!" * 3)
        args.overrides.extend([
            "train.epochs=2",
            "train.patience=2",
            "data.max_per_class=4" # smoke.max_per_class equivalent
        ])
        
    config = load_config(args.config, args.overrides)
    
    # Save resolved config
    def config_to_dict(c):
        if hasattr(c, "items"):
            return {k: config_to_dict(v) for k, v in c.items()}
        return c
    
    with open(out_dir / "config_resolved.yaml", "w") as f:
        yaml.dump(config_to_dict(config), f)
        
    # Env info
    env_info = get_env_info()
    with open(out_dir / "env_info.json", "w") as f:
        json.dump(env_info, f, indent=4)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, val_loader, test_loader, data_info = build_loaders(config, smoke=args.smoke)
    loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    class_names = data_info.class_names
    num_classes = data_info.K
    
    model_kwargs = config.get("model", {}).copy()
    model_kwargs.pop("name", None)
    model_kwargs.pop("variant", None)
    
    # Create Model
    model = build_model(
        name=config.model.name,
        variant=config.model.variant,
        num_classes=num_classes,
        **model_kwargs
    )
    model.to(device)
    
    param_count = count_parameters(model)[0]
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    
    train_size = len(loaders["train"].dataset)
    val_size = len(loaders["val"].dataset)
    test_size = len(loaders["test"].dataset)
    
    print_start_banner(run_name, config.model.name, param_count, device, train_size, val_size, test_size, num_classes, config)
    
    # Class weights
    class_weights = None
    if config.train.get("class_weights", False):
        class_weights = data_info.class_weights
        
    trainer = Trainer(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        config=config,
        device=device,
        class_names=class_names,
        out_dir=out_dir,
        class_weights=class_weights,
        resume_path=args.resume
    )
    
    train_metrics = trainer.fit()
    
    print("Training finished. Evaluating on validation split...")
    best_checkpoint = out_dir / "best.pt"
    if not best_checkpoint.exists():
        best_checkpoint = out_dir / "last.pt"
        
    val_metrics, val_confused, _ = evaluate(best_checkpoint, "val", out_dir)
    print("Evaluating on test split...")
    test_metrics, test_confused, test_report = evaluate(best_checkpoint, "test", out_dir)
    
    # Summary
    summary = {
        "run_name": run_name,
        "description": config.get("description", ""),
        "model": config.model.name,
        "variant": config.model.variant,
        "parameters": param_count,
        "size_mb": model_size_mb,
        "best_epoch": train_metrics["best_epoch"],
        "epochs_run": train_metrics["epochs_run"],
        "total_time_s": train_metrics["total_time_s"],
        "device": str(device),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=4)
        
    # NOTE THIS FOR PPT
    worst_classes = test_report.sort_values('f1').head(3)
    worst_classes_str = ", ".join([f"{row['class_name']} ({row['f1']:.2f})" for _, row in worst_classes.iterrows()])
    
    top3_confused = test_confused[:3]
    confused_str = ", ".join([f"{c['true_class']}->{c['predicted_class']} ({c['count']})" for c in top3_confused])
    
    note_block = f"""===== NOTE THIS FOR PPT =====
Run Name       : {run_name}
Model & Variant: {config.model.name} {config.model.variant}
Description    : {config.get('description', '')}
Parameters     : {param_count:,}
Model Size     : {model_size_mb:.2f} MB
Epochs Run     : {train_metrics['epochs_run']} (Best: {train_metrics['best_epoch']})
Total Time     : {train_metrics['total_time_s']/60:.1f} mins
Device         : {device}

--- Validation ---
Accuracy : {val_metrics['accuracy']:.4f}
Macro F1 : {val_metrics['macro_f1']:.4f}

--- Test ---
Accuracy : {test_metrics['accuracy']:.4f}
Top-3 Acc: {test_metrics.get('top3_accuracy', 0):.4f}
Macro P  : {test_metrics['macro_precision']:.4f}
Macro R  : {test_metrics['macro_recall']:.4f}
Macro F1 : {test_metrics['macro_f1']:.4f}
Weight F1: {test_metrics['weighted_f1']:.4f}

Worst F1 Classes: {worst_classes_str}
Most Confused   : {confused_str}
===== END NOTE ====="""

    print("\n" + note_block)
    
    with open(out_dir / "note_block.txt", "w") as f:
        f.write(note_block)
        
    if not args.smoke:
        append_to_results_log(
            run_name=run_name,
            model_name=config.model.name,
            variant=config.model.variant,
            description=config.get('description', ''),
            param_count=param_count,
            model_size_mb=model_size_mb,
            epochs_run=train_metrics['epochs_run'],
            best_epoch=train_metrics['best_epoch'],
            total_time_s=train_metrics['total_time_s'],
            device=str(device),
            val_acc=val_metrics['accuracy'],
            val_macro_f1=val_metrics['macro_f1'],
            test_acc=test_metrics['accuracy'],
            test_top3_acc=test_metrics.get('top3_accuracy', 0),
            test_macro_p=test_metrics['macro_precision'],
            test_macro_r=test_metrics['macro_recall'],
            test_macro_f1=test_metrics['macro_f1'],
            test_weighted_f1=test_metrics['weighted_f1'],
            note_block=note_block
        )

if __name__ == "__main__":
    main()
