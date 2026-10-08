import os
import time
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def get_model_size_mb(model: nn.Module) -> float:
    # Save state dict to a temp file and get size
    temp_file = Path("temp_model_size.pt")
    torch.save(model.state_dict(), temp_file)
    size_mb = temp_file.stat().st_size / (1024 * 1024)
    temp_file.unlink()
    return size_mb

def calculate_macs(model: nn.Module, input_size: tuple, device: torch.device) -> float:
    macs = 0
    hooks = []
    
    def conv_hook(module, input, output):
        nonlocal macs
        # input[0] shape: (batch_size, in_channels, in_h, in_w)
        # output shape: (batch_size, out_channels, out_h, out_w)
        out_c, out_h, out_w = output.shape[1], output.shape[2], output.shape[3]
        in_c = input[0].shape[1]
        k_h, k_w = module.kernel_size
        groups = module.groups
        # MACs per output element: (in_c / groups) * k_h * k_w
        # Total MACs: out_c * out_h * out_w * (in_c / groups) * k_h * k_w
        macs += out_c * out_h * out_w * (in_c // groups) * k_h * k_w

    def linear_hook(module, input, output):
        nonlocal macs
        # MACs: out_features * in_features
        macs += module.out_features * module.in_features
        
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
            
    # Dummy forward pass
    dummy_input = torch.randn(1, 3, *input_size).to(device)
    model.eval()
    with torch.no_grad():
        model(dummy_input)
        
    for hook in hooks:
        hook.remove()
        
    return macs / 1e6  # Return in millions

def time_forward_pass(model: nn.Module, input_size: tuple, device: torch.device, batch_size: int = 1, num_warmup: int = 30, num_runs: int = 200) -> dict:
    dummy_input = torch.randn(batch_size, 3, *input_size).to(device)
    model.eval()
    
    # Warmup
    with torch.inference_mode():
        for _ in range(num_warmup):
            model(dummy_input)
            
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    times = []
    with torch.inference_mode():
        for _ in range(num_runs):
            if device.type == 'cuda':
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                model(dummy_input)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end)) # in ms
            else:
                t0 = time.perf_counter()
                model(dummy_input)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000) # in ms
                
    times = np.array(times)
    return {
        'median_ms': float(np.median(times)),
        'mean_ms': float(np.mean(times)),
        'p95_ms': float(np.percentile(times, 95)),
        'images_per_sec': float(batch_size / (np.median(times) / 1000.0))
    }

def time_full_pipeline(model: nn.Module, test_crops: list, transform, device: torch.device, num_warmup: int = 30, num_runs: int = 200) -> dict:
    from PIL import Image
    model.eval()
    
    # Sample a real crop
    sample_path = test_crops[0]
    
    times = []
    
    # Warmup
    for _ in range(num_warmup):
        img = Image.open(sample_path).convert('RGB')
        t_img = transform(img).unsqueeze(0).to(device)
        with torch.inference_mode():
            model(t_img)
            
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    for i in range(num_runs):
        crop_path = test_crops[i % len(test_crops)]
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            
            img = Image.open(crop_path).convert('RGB')
            t_img = transform(img).unsqueeze(0).to(device)
            with torch.inference_mode():
                model(t_img)
                
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            img = Image.open(crop_path).convert('RGB')
            t_img = transform(img).unsqueeze(0).to(device)
            with torch.inference_mode():
                model(t_img)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
            
    times = np.array(times)
    return {
        'median_ms': float(np.median(times)),
        'mean_ms': float(np.mean(times)),
        'p95_ms': float(np.percentile(times, 95)),
        'images_per_sec': float(1.0 / (np.median(times) / 1000.0))
    }
