import time
import csv
import json
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from ml.engine.metrics import compute_classification_metrics
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

def load_checkpoint(path, device="cpu"):
    """
    Rebuilds the model from the checkpoint alone.
    """
    from ml.models.factory import build_model
    checkpoint = torch.load(path, map_location=device)
    config_dict = checkpoint.get("config", {})
    from ml.config import Config
    config = Config(config_dict)
    
    model_kwargs = config.get("model", {}).copy()
    model_kwargs.pop("name", None)
    model_kwargs.pop("variant", None)
    
    model = build_model(
        name=checkpoint["model_name"],
        variant=checkpoint["variant"],
        num_classes=checkpoint["num_classes"],
        **model_kwargs
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    
    return model, checkpoint

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        config,
        device,
        class_names,
        out_dir,
        class_weights=None,
        resume_path=None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        self.epochs = self.config.train.get("epochs")
        self.patience = self.config.train.get("early_stopping_patience", 5)
        self.grad_clip = self.config.train.get("grad_clip") or 1.0
        amp_val = self.config.train.get("amp", "auto")
        if str(amp_val).lower() == "auto":
            self.mixed_precision = self.device.type == "cuda"
        elif str(amp_val).lower() == "true":
            self.mixed_precision = True
        else:
            self.mixed_precision = False
            
        self.channels_last = self.config.train.get("channels_last", True) and self.device.type == "cuda"
        
        if self.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
            
        # Loss
        loss_kwargs = {}
        if class_weights is not None:
            loss_kwargs["weight"] = class_weights.to(self.device)
        ls = self.config.train.get("label_smoothing") or 0.0
        if ls > 0:
            loss_kwargs["label_smoothing"] = ls
        self.criterion = nn.CrossEntropyLoss(**loss_kwargs)
        
        # Optimiser
        # Support parameter groups if model provides a method for it
        if hasattr(self.model, "get_parameter_groups"):
            param_groups = self.model.get_parameter_groups(
                lr=self.config.train.get("lr", 0.001),
                weight_decay=self.config.train.get("weight_decay", 0.0)
            )
            self.optimizer = AdamW(param_groups)
        else:
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.config.train.get("lr", 0.001),
                weight_decay=self.config.train.get("weight_decay", 0.0)
            )
            
        # Scheduler: linear warm-up then cosine annealing, stepped once per epoch
        warmup_epochs = self.config.train.get("warmup_epochs")
        if warmup_epochs is None:
            warmup_epochs = min(5, self.epochs // 10)
        if warmup_epochs > 0:
            warmup_scheduler = LinearLR(self.optimizer, start_factor=0.01, total_iters=warmup_epochs)
            cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs - warmup_epochs)
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs]
            )
        else:
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs)
            
        if self.mixed_precision:
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            # We need a dummy scaler for CPU or disabled amp
            self.scaler = torch.amp.GradScaler('cuda', enabled=False)
        
        self.start_epoch = 0
        self.best_metric = 0.0
        self.epochs_without_improvement = 0
        self.history = []
        
        if resume_path and Path(resume_path).exists():
            self._load_resume_checkpoint(resume_path)
            
        self.history_file = self.out_dir / "history.csv"
        if not self.history_file.exists() or self.start_epoch == 0:
            with open(self.history_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_macro_f1", "lr", "epoch_time_s"])
                
    def _load_resume_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scheduler_state" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        if "scaler_state" in checkpoint and self.mixed_precision:
            self.scaler.load_state_dict(checkpoint["scaler_state"])
            
        self.start_epoch = checkpoint["epoch"] + 1
        self.best_metric = checkpoint.get("best_metric", 0.0)
        self.epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)
        print(f"Resumed from epoch {self.start_epoch} (best metric: {self.best_metric:.4f})")
        
    def _save_checkpoint(self, path, is_best=False):
        # Convert config object to dict for saving
        def config_to_dict(c):
            if isinstance(c, dict) or type(c).__name__ == 'Config':
                return {k: config_to_dict(v) for k, v in c.items()}
            return c
            
        state = {
            "epoch": self.start_epoch, # Will be set to current epoch before call
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict() if self.mixed_precision else None,
            "best_metric": self.best_metric,
            "epochs_without_improvement": self.epochs_without_improvement,
            "model_name": self.config.model.get("name", "unknown"),
            "variant": self.config.model.get("variant", "unknown"),
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "img_size": self.config.get("data", {}).get("img_size", 64),
            "norm_mean": self.config.get("data", {}).get("norm_mean", [0.5, 0.5, 0.5]),
            "norm_std": self.config.get("data", {}).get("norm_std", [0.5, 0.5, 0.5]),
            "config": config_to_dict(self.config)
        }
        torch.save(state, path)
        if is_best:
            best_path = self.out_dir / "best.pt"
            torch.save(state, best_path)

    def set_frozen_batchnorm_to_eval(self):
        """Ensure BatchNorm layers in frozen blocks stay in eval mode."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                # If all parameters in this BN are frozen, set to eval
                if all(not p.requires_grad for p in module.parameters()):
                    module.eval()

    def train_epoch(self):
        self.model.train()
        self.set_frozen_batchnorm_to_eval()
        
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(self.train_loader, desc="Train", leave=False)
        for images, targets, _ in pbar:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            if self.channels_last:
                images = images.to(memory_format=torch.channels_last)
                
            self.optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type=self.device.type, enabled=bool(self.mixed_precision)):
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item() * images.size(0)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        metrics = compute_classification_metrics(all_targets, all_preds)
        avg_loss = total_loss / len(self.train_loader.dataset)
        return avg_loss, metrics["accuracy"]
        
    def val_epoch(self):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(self.val_loader, desc="Val", leave=False)
        with torch.no_grad():
            for images, targets, _ in pbar:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                if self.channels_last:
                    images = images.to(memory_format=torch.channels_last)
                    
                with torch.amp.autocast(device_type=self.device.type, enabled=bool(self.mixed_precision)):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
                    
                total_loss += loss.item() * images.size(0)
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
        metrics = compute_classification_metrics(all_targets, all_preds)
        avg_loss = total_loss / len(self.val_loader.dataset)
        return avg_loss, metrics["accuracy"], metrics["macro_f1"]

    def fit(self):
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            
        start_time = time.time()
        epoch_times = []
        best_epoch = 0
        
        print(f"Training started on {self.device}")
        
        try:
            for epoch in range(self.start_epoch, self.epochs):
                self.start_epoch = epoch # Update for checkpoint saving
                epoch_start_time = time.time()
                
                # Get current LR (assume first param group)
                current_lr = self.optimizer.param_groups[0]["lr"]
                
                # Train and Val
                train_loss, train_acc = self.train_epoch()
                val_loss, val_acc, val_macro_f1 = self.val_epoch()
                
                self.scheduler.step()
                
                epoch_time = time.time() - epoch_start_time
                epoch_times.append(epoch_time)
                
                # Logging
                print(f"Epoch {epoch+1:03d}/{self.epochs:03d} | "
                      f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} MacroF1: {val_macro_f1:.4f} | "
                      f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
                      
                with open(self.history_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch+1, train_loss, train_acc, val_loss, val_acc, val_macro_f1, current_lr, epoch_time])
                    
                # Model saving & Early stopping
                is_best = val_macro_f1 > self.best_metric
                if is_best:
                    self.best_metric = val_macro_f1
                    self.epochs_without_improvement = 0
                    best_epoch = epoch + 1
                else:
                    self.epochs_without_improvement += 1
                    
                self._save_checkpoint(self.out_dir / "last.pt", is_best=is_best)
                
                if self.epochs_without_improvement >= self.patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break
                    
        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Saving last.pt...")
            self._save_checkpoint(self.out_dir / "last.pt")
            
        total_time = time.time() - start_time
        avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0.0
        
        peak_memory_mb = 0
        if self.device.type == "cuda":
            peak_memory_mb = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
            
        return {
            "total_time_s": total_time,
            "avg_epoch_time_s": avg_epoch_time,
            "best_epoch": best_epoch,
            "best_val_macro_f1": self.best_metric,
            "epochs_run": len(epoch_times),
            "peak_memory_mb": peak_memory_mb
        }
