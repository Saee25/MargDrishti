import os
import sys
import json
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import seaborn as sns
from torchvision.transforms import v2

from ml.engine.stats import agreement_table, mcnemar_test, bootstrap_confidence_intervals, group_by_crop_size
from ml.explain.gradcam import GradCAM, apply_colormap_on_image
from ml.models.factory import build_model

# Colors from design language
COLORS = {
    'margnet': '#7C5FA6', # purple-500
    'resnet_ft': '#C39A45', # ochre
    'resnet_frozen': '#B3ACDF', # lavender-300
    'sage': '#6F9E86',
    'rose': '#C0707F',
    'ink': '#2A2438'
}

def load_summary(path):
    with open(path, 'r') as f:
        return json.load(f)

def load_predictions(path):
    return pd.read_csv(path)

def generate_comparison(is_smoke=False):
    out_dir = Path("reports/comparison" if not is_smoke else "reports/_smoke_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if benchmark is run
    bench_path = out_dir / "benchmark.json"
    if not bench_path.exists():
        if not is_smoke:
            print("Benchmark results not found. Please run scripts.benchmark first.")
            return
        else:
            # mock for smoke
            bench_dict = {}
    else:
        with open(bench_path, 'r') as f:
            bench_data = json.load(f)
        bench_dict = {b['model']: b for b in bench_data}
        
    prefix = "_smoke_" if is_smoke else ""
    marg_sum = load_summary(f"experiments/{prefix}custom_v5_margnet/summary.json")
    rf_sum = load_summary(f"experiments/{prefix}resnet50_frozen/summary.json")
    rft_sum = load_summary(f"experiments/{prefix}resnet50_finetune/summary.json")
    
    marg_pred = load_predictions(f"experiments/{prefix}custom_v5_margnet/predictions.csv")
    rf_pred = load_predictions(f"experiments/{prefix}resnet50_frozen/predictions.csv")
    rft_pred = load_predictions(f"experiments/{prefix}resnet50_finetune/predictions.csv")
    
    # Assert sets are identical
    assert all(marg_pred['crop_path'] == rft_pred['crop_path'])
    
    y_true = marg_pred['true_class'].values
    y_p_marg = marg_pred['pred_class'].values
    y_p_rft = rft_pred['pred_class'].values
    y_p_rf = rf_pred['pred_class'].values
    
    # Headline Table
    def get_row(name, sum_data, bench_model_name, is_pretrain, epochs_run, best_epoch):
        b = bench_dict.get(bench_model_name, {})
        metrics = sum_data.get('metrics', {})
        return {
            'Model': name,
            'Input Size': '224x224' if 'ResNet' in name else '64x64',
            'Pretrained': 'Yes' if is_pretrain else 'No',
            'Parameters': b.get('params', 0),
            'Trainable Params': b.get('trainable_params', 0),
            'Size (MB)': b.get('size_mb', 0.0),
            'MACs (M)': b.get('macs_m', 0.0),
            'Epochs Run': epochs_run,
            'Best Epoch': best_epoch,
            'Training Time': sum_data.get('total_time', sum_data.get('training_time_seconds', 0)) / 60.0,
            'Device': sum_data.get('device', 'GPU'),
            'Test Accuracy': metrics.get('test_accuracy', 0.0),
            'Top-3 Accuracy': metrics.get('test_top3_accuracy', 0.0),
            'Macro Precision': metrics.get('test_macro_precision', 0.0),
            'Macro Recall': metrics.get('test_macro_recall', 0.0),
            'Macro F1': metrics.get('test_macro_f1', 0.0),
            'Weighted F1': metrics.get('test_weighted_f1', 0.0),
            'CPU Latency (ms)': b.get('pipe_bs1_median_ms', 0.0),
            'CPU Img/sec': b.get('fwd_bs32_img_sec', 0.0)
        }
        
    rows = [
        get_row('MargNet', marg_sum, 'MargNet', False, 40, marg_sum.get('best_epoch', 0)),
        get_row('ResNet50 frozen', rf_sum, 'ResNet50 frozen', True, 20, rf_sum.get('best_epoch', 0)),
        get_row('ResNet50 fine-tuned', rft_sum, 'ResNet50 fine-tuned', True, 20, rft_sum.get('best_epoch', 0))
    ]
    
    df_head = pd.DataFrame(rows)
    
    # Ratios against ResNet50 fine-tuned
    rft_row = df_head[df_head['Model'] == 'ResNet50 fine-tuned'].iloc[0]
    marg_row = df_head[df_head['Model'] == 'MargNet'].iloc[0]
    
    times_fewer_params = rft_row['Parameters'] / marg_row['Parameters'] if marg_row['Parameters'] > 0 else 0
    times_smaller = rft_row['Size (MB)'] / marg_row['Size (MB)'] if marg_row['Size (MB)'] > 0 else 0
    times_faster_latency = rft_row['CPU Latency (ms)'] / marg_row['CPU Latency (ms)'] if marg_row['CPU Latency (ms)'] > 0 else 0
    acc_gap = rft_row['Test Accuracy'] - marg_row['Test Accuracy']
    
    df_head['Times Fewer Params'] = df_head.apply(lambda r: rft_row['Parameters'] / r['Parameters'] if r['Parameters'] > 0 else 0, axis=1)
    df_head['Times Smaller'] = df_head.apply(lambda r: rft_row['Size (MB)'] / r['Size (MB)'] if r['Size (MB)'] > 0 else 0, axis=1)
    df_head['Times Faster (Latency)'] = df_head.apply(lambda r: rft_row['CPU Latency (ms)'] / r['CPU Latency (ms)'] if r['CPU Latency (ms)'] > 0 else 0, axis=1)
    df_head['Acc Gap (pp)'] = df_head.apply(lambda r: rft_row['Test Accuracy'] - r['Test Accuracy'], axis=1)
    
    df_head.to_csv(out_dir / "comparison_table.csv", index=False)
    df_head.to_json(out_dir / "comparison_table.json", orient='records', indent=2)
    df_head.to_markdown(out_dir / "comparison_table.md", index=False)
    
    # Agreement Table
    bc, o1, o2, bw = agreement_table(y_true, y_p_marg, y_p_rft)
    total = len(y_true)
    
    pval, pval_msg = mcnemar_test(o1, o2)
    ci = bootstrap_confidence_intervals(y_true, y_p_marg, y_p_rft)
    
    # Per-class analysis
    with open("data/processed/class_map.json", 'r') as f:
        class_map = json.load(f)
    rev_class_map = {int(v): k for k, v in class_map.items()}
    
    manifest = pd.read_csv("data/processed/manifest.csv")
    train_counts = manifest[manifest['split'] == 'train']['class_id'].value_counts().to_dict()
    
    from sklearn.metrics import classification_report
    cr_marg = classification_report(y_true, y_p_marg, output_dict=True, zero_division=0)
    cr_rft = classification_report(y_true, y_p_rft, output_dict=True, zero_division=0)
    
    pc_data = []
    for cls_id in np.unique(y_true):
        cls_str = str(cls_id)
        f1_marg = cr_marg[cls_str]['f1-score'] if cls_str in cr_marg else 0
        f1_rft = cr_rft[cls_str]['f1-score'] if cls_str in cr_rft else 0
        pc_data.append({
            'class_id': cls_id,
            'class_name': rev_class_map.get(cls_id, f"Class {cls_id}"),
            'test_support': cr_marg[cls_str]['support'] if cls_str in cr_marg else 0,
            'train_count': train_counts.get(cls_id, 0),
            'f1_margnet': f1_marg,
            'f1_resnet_ft': f1_rft,
            'difference': f1_marg - f1_rft
        })
    df_pc = pd.DataFrame(pc_data)
    df_pc.to_csv(out_dir / "per_class_comparison.csv", index=False)
    
    df_pc_sorted = df_pc.sort_values('difference')
    plt.figure(figsize=(10, 15))
    colors = [COLORS['resnet_ft'] if val < 0 else COLORS['margnet'] for val in df_pc_sorted['difference']]
    sns.barplot(data=df_pc_sorted, x='difference', y='class_name', palette=colors)
    plt.title('F1 Difference: MargNet vs ResNet50 fine-tuned')
    plt.xlabel('Difference (MargNet - ResNet50 FT)')
    plt.tight_layout()
    plt.savefig(out_dir / "per_class_f1_difference.png", dpi=200)
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.scatter(df_pc['train_count'], df_pc['f1_margnet'], color=COLORS['margnet'], alpha=0.6, label='MargNet')
    plt.scatter(df_pc['train_count'], df_pc['f1_resnet_ft'], color=COLORS['resnet_ft'], alpha=0.6, label='ResNet50 FT')
    plt.xscale('log')
    plt.xlabel('Training Count (Log Scale)')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Number of Training Crops')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "f1_vs_train_count.png", dpi=200)
    plt.close()
    
    marg_wins = df_pc[df_pc['difference'] > 0]['class_name'].tolist()
    rft_wins_big = df_pc[df_pc['difference'] < -0.05]['class_name'].tolist()
    
    # Crop size analysis
    def get_shorter_side(path):
        with Image.open(path) as img:
            return min(img.size)
            
    test_crops = marg_pred['crop_path'].tolist()
    shorter_sides = np.array([get_shorter_side(p) for p in test_crops])
    crop_size_acc = group_by_crop_size(y_true, y_p_marg, y_p_rft, shorter_sides)
    
    df_size = pd.DataFrame(crop_size_acc).T
    df_size.to_csv(out_dir / "accuracy_by_crop_size.csv")
    
    df_size[['acc1', 'acc2']].plot(kind='bar', color=[COLORS['margnet'], COLORS['resnet_ft']], figsize=(8,5))
    plt.title("Accuracy by Crop Size (Terciles)")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=0)
    plt.legend(["MargNet", "ResNet50 FT"])
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_by_crop_size.png", dpi=200)
    plt.close()
    
    # Confusion side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    cm_m = np.load(f"experiments/{prefix}custom_v5_margnet/confusion_matrix.npy")
    cm_r = np.load(f"experiments/{prefix}resnet50_finetune/confusion_matrix.npy")
    sns.heatmap(cm_m, ax=axes[0], cmap="Purples", cbar=False, xticklabels=False, yticklabels=False)
    axes[0].set_title("MargNet Confusion Matrix")
    sns.heatmap(cm_r, ax=axes[1], cmap="YlOrBr", cbar=False, xticklabels=False, yticklabels=False)
    axes[1].set_title("ResNet50 FT Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_side_by_side.png", dpi=200)
    plt.close()
    
    # Efficiency charts
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    models = ['MargNet', 'ResNet50\nFrozen', 'ResNet50\nFT']
    params = [df_head.iloc[0]['Parameters']/1e6, df_head.iloc[1]['Parameters']/1e6, df_head.iloc[2]['Parameters']/1e6]
    sizes = [df_head.iloc[0]['Size (MB)'], df_head.iloc[1]['Size (MB)'], df_head.iloc[2]['Size (MB)']]
    latencies = [df_head.iloc[0]['CPU Latency (ms)'], df_head.iloc[1]['CPU Latency (ms)'], df_head.iloc[2]['CPU Latency (ms)']]
    
    colors_eff = [COLORS['margnet'], COLORS['resnet_frozen'], COLORS['resnet_ft']]
    
    axes[0].bar(models, params, color=colors_eff)
    axes[0].set_title('Parameters (Millions)')
    axes[0].set_yscale('log')
    
    axes[1].bar(models, sizes, color=colors_eff)
    axes[1].set_title('Model Size (MB)')
    axes[1].set_yscale('log')
    
    axes[2].bar(models, latencies, color=colors_eff)
    axes[2].set_title('CPU Latency (ms)')
    
    plt.tight_layout()
    plt.savefig(out_dir / "efficiency_bars.png", dpi=200)
    plt.close()

    # Explainability: Grad-CAM
    np.random.seed(42)
    both_correct_idx = np.where((y_p_marg == y_true) & (y_p_rft == y_true))[0]
    marg_only_idx = np.where((y_p_marg == y_true) & (y_p_rft != y_true))[0]
    rft_only_idx = np.where((y_p_marg != y_true) & (y_p_rft == y_true))[0]
    
    sel_bc = np.random.choice(both_correct_idx, min(4, len(both_correct_idx)), replace=False)
    sel_mo = np.random.choice(marg_only_idx, min(2, len(marg_only_idx)), replace=False)
    sel_ro = np.random.choice(rft_only_idx, min(2, len(rft_only_idx)), replace=False)
    
    selected_indices = np.concatenate([sel_bc, sel_mo, sel_ro])
    
    import yaml
    with open(f"experiments/{prefix}custom_v5_margnet/config_resolved.yaml", 'r') as f:
        cfg_m = yaml.safe_load(f)
    ckpt_m = torch.load(f"experiments/{prefix}custom_v5_margnet/best.pt", map_location='cpu')
    model_m = build_model(cfg_m['model']['name'], cfg_m['model'].get('variant', 'v5'), num_classes=ckpt_m.get('num_classes', 75))
    model_m.load_state_dict(ckpt_m.get('model_state', ckpt_m))
    
    with open(f"experiments/{prefix}resnet50_finetune/config_resolved.yaml", 'r') as f:
        cfg_r = yaml.safe_load(f)
    ckpt_r = torch.load(f"experiments/{prefix}resnet50_finetune/best.pt", map_location='cpu')
    model_r = build_model(cfg_r['model']['name'], cfg_r['model'].get('variant', 'finetune'), num_classes=ckpt_r.get('num_classes', 75))
    model_r.load_state_dict(ckpt_r.get('model_state', ckpt_r))
    
    cam_m = GradCAM(model_m)
    cam_r = GradCAM(model_r)
    
    t_m = v2.Compose([v2.Resize((64, 64)), v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=(0.334, 0.312, 0.320), std=(0.245, 0.239, 0.247))])
    t_r = v2.Compose([v2.Resize((224, 224)), v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    
    fig, axes = plt.subplots(len(selected_indices), 3, figsize=(10, 3 * len(selected_indices)))
    
    for i, idx in enumerate(selected_indices):
        path = test_crops[idx]
        img = Image.open(path).convert('RGB')
        
        hm_m = cam_m.generate(t_m(img).unsqueeze(0), y_true[idx])
        img_m = apply_colormap_on_image(img.resize((64, 64)), hm_m)
        
        hm_r = cam_r.generate(t_r(img).unsqueeze(0), y_true[idx])
        img_r = apply_colormap_on_image(img.resize((224, 224)), hm_r)
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"True: {rev_class_map[y_true[idx]]}")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(img_m)
        axes[i, 1].set_title(f"MargNet: {rev_class_map[y_p_marg[idx]]}")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(img_r)
        axes[i, 2].set_title(f"ResNet FT: {rev_class_map[y_p_rft[idx]]}")
        axes[i, 2].axis('off')
        
    plt.tight_layout()
    plt.savefig(out_dir / "gradcam_comparison.png", dpi=200)
    plt.close()
    
    cam_m.remove_hooks()
    cam_r.remove_hooks()

    # Note block
    note = (
        "===== NOTE THIS FOR PPT =====\n"
        f"Headline Table:\n{df_head[['Model', 'Test Accuracy', 'Parameters', 'CPU Latency (ms)']].to_string()}\n\n"
        f"Ratios: MargNet has {times_fewer_params:.1f}x fewer params, is {times_smaller:.1f}x smaller, "
        f"and {times_faster_latency:.1f}x faster than ResNet50 FT.\n"
        f"Accuracy Gap: {acc_gap*100:.1f} percentage points.\n\n"
        f"Agreement: Both correct {bc}, MargNet only {o1}, ResNet FT only {o2}, Both wrong {bw}.\n"
        f"McNemar p-value: {pval:.4f}. {pval_msg}\n\n"
        f"Bootstrap 95% CIs:\n"
        f"MargNet Acc: [{ci['model1_acc'][0]:.4f}, {ci['model1_acc'][1]:.4f}]\n"
        f"ResNet FT Acc: [{ci['model2_acc'][0]:.4f}, {ci['model2_acc'][1]:.4f}]\n"
        f"Diff Acc: [{ci['acc_diff'][0]:.4f}, {ci['acc_diff'][1]:.4f}]\n\n"
        f"MargNet wins {len(marg_wins)} classes, ResNet FT wins heavily on {len(rft_wins_big)} classes.\n"
        "===== END NOTE ====="
    )
    with open(out_dir / "comparison_report.md", 'w') as f:
        f.write("# Detailed Comparison\n\n" + note)
        
    print(note)

if __name__ == '__main__':
    is_smoke = '--smoke' in sys.argv
    generate_comparison(is_smoke=is_smoke)
