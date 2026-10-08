import os
import json
import torch
import pandas as pd
from pathlib import Path
from ml.engine.benchmark import count_parameters, get_model_size_mb, calculate_macs, time_forward_pass, time_full_pipeline
from ml.models.factory import build_model

from torchvision.transforms import v2

def get_test_crops():
    manifest_path = Path("data/processed/manifest.csv")
    df = pd.read_csv(manifest_path)
    test_df = df[df['split'] == 'test']
    crops_dir = Path("data/processed/crops")
    return [str(crops_dir / p) for p in test_df['crop_path'].tolist()]

def main():
    print("================================================================")
    print("Please close heavy apps (browsers with many tabs, games, etc.),")
    print("keep the laptop on power, and do not touch it while this runs.")
    print("================================================================\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking on device: {device}")
    
    test_crops = get_test_crops()
    if not test_crops:
        print("No test crops found!")
        return
        
    models_to_test = [
        ('MargNet', 'experiments/custom_v5_margnet/config_resolved.yaml', 'experiments/custom_v5_margnet/best.pt', 64, (0.334, 0.312, 0.320), (0.245, 0.239, 0.247)), # these mean/std might be config-dependent, I'll rely on the standard transforms for benchmark
        ('ResNet50 frozen', 'experiments/resnet50_frozen/config_resolved.yaml', 'experiments/resnet50_frozen/best.pt', 224, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ('ResNet50 fine-tuned', 'experiments/resnet50_finetune/config_resolved.yaml', 'experiments/resnet50_finetune/best.pt', 224, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
    
    import yaml
    
    results = []
    
    for name, config_path, weights_path, img_size, mean, std in models_to_test:
        print(f"\nBenchmarking {name}...")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        checkpoint = torch.load(weights_path, map_location='cpu')
        num_classes = checkpoint.get('num_classes', config.get('model', {}).get('num_classes', 75))
        model = build_model(config['model']['name'], config['model'].get('variant', 'v1'), num_classes=num_classes)
        model.load_state_dict(checkpoint.get('model_state', checkpoint))
        model.to(device)
        model.eval()
        
        params = count_parameters(model)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        size_mb = get_model_size_mb(model)
        macs_m = calculate_macs(model, (img_size, img_size), device)
        
        transforms = v2.Compose([
            v2.Resize((img_size, img_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std)
        ])
        
        print("  Warming up and timing forward pass (bs=1)...")
        fwd_bs1 = time_forward_pass(model, (img_size, img_size), device, batch_size=1)
        
        print("  Warming up and timing full pipeline (bs=1)...")
        pipe_bs1 = time_full_pipeline(model, test_crops, transforms, device)
        
        print("  Warming up and timing forward pass (bs=32)...")
        fwd_bs32 = time_forward_pass(model, (img_size, img_size), device, batch_size=32)
        
        res = {
            'model': name,
            'params': params,
            'trainable_params': trainable_params,
            'size_mb': size_mb,
            'macs_m': macs_m,
            'fwd_bs1_median_ms': fwd_bs1['median_ms'],
            'fwd_bs1_p95_ms': fwd_bs1['p95_ms'],
            'fwd_bs1_img_sec': fwd_bs1['images_per_sec'],
            'pipe_bs1_median_ms': pipe_bs1['median_ms'],
            'pipe_bs1_img_sec': pipe_bs1['images_per_sec'],
            'fwd_bs32_median_ms': fwd_bs32['median_ms'],
            'fwd_bs32_img_sec': fwd_bs32['images_per_sec'],
            'device': str(device)
        }
        results.append(res)
        
    out_dir = Path("reports/comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(results)
    df.to_csv(out_dir / "benchmark.csv", index=False)
    
    with open(out_dir / "benchmark.json", 'w') as f:
        json.dump(results, f, indent=2)
        
    print("\n===== NOTE THIS FOR PPT =====")
    print(f"{'Model':<25} | {'Params':>10} | {'Size (MB)':>9} | {'MACs (M)':>8} | {'Lat. pipe (ms)':>14} | {'Thru BS32 (img/s)':>17}")
    print("-" * 95)
    for r in results:
        print(f"{r['model']:<25} | {r['params']:>10,} | {r['size_mb']:>9.2f} | {r['macs_m']:>8.2f} | {r['pipe_bs1_median_ms']:>14.2f} | {r['fwd_bs32_img_sec']:>17.2f}")
    print("===== END NOTE =====")

if __name__ == '__main__':
    main()
