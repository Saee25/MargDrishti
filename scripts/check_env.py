import sys
import platform
import psutil
from pathlib import Path
from ml.utils.device import get_device, describe_device
from ml.utils.results import print_note_block, append_results_log
from ml.config import get_project_root
import torch
import torchvision

def main():
    root = get_project_root()
    device = get_device()
    
    py_version = platform.python_version()
    os_name = platform.system()
    os_release = platform.release()
    torch_version = torch.__version__
    tv_version = torchvision.__version__
    
    device_desc = describe_device()
    cpu_cores = psutil.cpu_count(logical=True)
    ram_gb = psutil.virtual_memory().total / (1024**3)
    
    # Check data/raw
    raw_dir = root / "data" / "raw"
    has_data = False
    if raw_dir.exists() and any(raw_dir.iterdir()):
        for f in raw_dir.iterdir():
            if f.name != ".gitkeep":
                has_data = True
                break
                
    # Tiny tensor operation
    try:
        t = torch.tensor([1.0, 2.0], device=device)
        t = t * 2
        tensor_status = "Success"
    except Exception as e:
        tensor_status = f"Failed: {str(e)}"
        
    print(f"Python: {py_version}")
    print(f"OS: {os_name} {os_release}")
    print(f"PyTorch: {torch_version}, Torchvision: {tv_version}")
    print(f"Device: {device_desc}")
    print(f"CPU Cores: {cpu_cores}")
    print(f"RAM: {ram_gb:.1f} GB")
    print(f"Data in data/raw: {'Yes' if has_data else 'No'}")
    print(f"Tensor op on {device}: {tensor_status}")
    
    rows = [
        f"Python: {py_version}",
        f"OS: {os_name} {os_release}",
        f"PyTorch: {torch_version}",
        f"Device: {device_desc}",
        f"CPU Cores: {cpu_cores}",
        f"System RAM: {ram_gb:.1f} GB"
    ]
    
    print_note_block("Hardware and Environment", rows)
    append_results_log("env_check", "Hardware and Environment", rows)

if __name__ == '__main__':
    main()
