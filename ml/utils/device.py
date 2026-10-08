import torch

def get_device() -> torch.device:
    """
    Returns the best available device: cuda, then mps, then cpu.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def describe_device() -> str:
    """
    Returns a string describing the device and GPU memory if available.
    """
    device = get_device()
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        return f"{name} ({mem:.1f} GB)"
    elif device.type == "mps":
        return "Apple Silicon (MPS)"
    else:
        return "CPU"
