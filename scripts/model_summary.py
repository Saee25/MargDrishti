import time
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torchinfo import summary
from ml.models.factory import build_model, count_parameters, model_size_mb
from pathlib import Path

def draw_architecture(model_name, describe_list, save_path):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    
    x = 0
    box_w = 2.5
    box_h = 1.2
    space = 0.6
    
    # Project palette
    bg_color = "#EEEDFB"  # lavender-100
    border_color = "#9479B8"  # purple-400
    text_color = "#2A2438"  # ink-900
    
    for i, layer in enumerate(describe_list):
        rect = plt.Rectangle((x, 0), box_w, box_h, facecolor=bg_color, edgecolor=border_color, lw=2, zorder=2)
        ax.add_patch(rect)
        
        name_text = f"{layer['name']}\n{layer['output_shape']}"
        ax.text(x + box_w/2, box_h/2, name_text, ha='center', va='center', color=text_color, fontsize=10, zorder=3)
        
        if i < len(describe_list) - 1:
            ax.arrow(x + box_w, box_h/2, space - 0.1, 0, head_width=0.1, head_length=0.1, fc=border_color, ec=border_color, zorder=1)
        
        x += box_w + space
        
    ax.set_xlim(-0.5, x + 0.5)
    ax.set_ylim(-0.5, 2.0)
    plt.title(f"{model_name} Architecture", fontsize=16, color=text_color, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

def main():
    reports_dir = Path("reports")
    summary_dir = reports_dir / "model_summaries"
    fig_dir = reports_dir / "figures" / "models"
    
    summary_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cpu')
    num_classes = 75
    
    variants = [
        ("v1", "custom_v1_plain"),
        ("v2", "custom_v2_bn"),
        ("v3", "custom_v3_aug"),
        ("v4", "custom_v4_weighted"),
        ("v5", "custom_v5_margnet")
    ]
    
    results = []
    
    for variant, name in variants:
        model = build_model("custom", variant, num_classes)
        model.eval()
        model.to(device)
        
        # 1. torchinfo summary
        with open(summary_dir / f"{variant}.txt", "w", encoding="utf-8") as f:
            model_stats = summary(model, input_size=(1, 3, 64, 64), verbose=0)
            f.write(str(model_stats))
            
        # 2. describe()
        desc = model.describe()
        with open(summary_dir / f"{variant}_layers.json", "w", encoding="utf-8") as f:
            json.dump(desc, f, indent=2)
            
        df_desc = pd.DataFrame(desc)
        with open(summary_dir / f"{variant}_layers.md", "w", encoding="utf-8") as f:
            f.write(df_desc.to_markdown(index=False))
            
        # 3. figures
        if variant == "v1":
            draw_architecture("PlainCNN", desc, fig_dir / "plaincnn_architecture.png")
        elif variant == "v5":
            draw_architecture("MargNet", desc, fig_dir / "margnet_architecture.png")
            
        # 4. info for table
        total, trainable = count_parameters(model)
        size_mb = model_size_mb(model)
        
        # Output shape
        dummy_in = torch.randn(1, 3, 64, 64)
        out = model(dummy_in)
        out_shape = tuple(out.shape)
        
        # Timing
        # Warmup
        for _ in range(10):
            model(dummy_in)
            
        t0 = time.time()
        n_iters = 100
        with torch.no_grad():
            for _ in range(n_iters):
                model(dummy_in)
        t1 = time.time()
        ms_per_image = ((t1 - t0) / n_iters) * 1000
        
        results.append({
            "Variant": variant,
            "Total Params": f"{total:,}",
            "Trainable Params": f"{trainable:,}",
            "Size (MB)": f"{size_mb:.2f}",
            "Output Shape": str(out_shape),
            "Fwd Time CPU (ms)": f"{ms_per_image:.2f}"
        })

    df_results = pd.DataFrame(results)
    print("===== NOTE THIS FOR PPT =====")
    print(df_results.to_markdown(index=False))
    print("===== END NOTE =====")

if __name__ == "__main__":
    main()
