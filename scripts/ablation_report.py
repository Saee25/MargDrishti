import argparse
import json
import csv
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Generate ablation report.")
    parser.add_argument("--smoke", action="store_true", help="Read smoke run folders instead of real ones.")
    args = parser.parse_args()

    steps = [
        {"run": "custom_v1_plain", "desc": "Baseline CNN (V1)"},
        {"run": "custom_v2_bn", "desc": "+ BatchNorm (V2)"},
        {"run": "custom_v3_aug", "desc": "+ Augmentation (V3)"},
        {"run": "custom_v4_weighted", "desc": "+ Class Weights (V4)"},
        {"run": "custom_v5_margnet", "desc": "+ MargNet (V5, Label Smoothing)"},
    ]

    prefix = "_smoke_" if args.smoke else ""
    out_dir = Path("reports/ablation") / ("_smoke" if args.smoke else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Palette
    purple_500 = "#7C5FA6"
    lavender_200 = "#CDCBF0"
    
    rows = []
    missing_runs = []
    
    prev_test_acc = 0.0
    prev_test_f1 = 0.0
    
    history_dfs = {}
    
    for step in steps:
        run_name = prefix + step["run"]
        run_dir = Path("experiments") / run_name
        
        summary_path = run_dir / "summary.json"
        history_path = run_dir / "history.csv"
        
        if not summary_path.exists():
            missing_runs.append(run_name)
            continue
            
        with open(summary_path, "r") as f:
            summary = json.load(f)
            
        if history_path.exists():
            history_dfs[step["run"]] = pd.read_csv(history_path)
            
        best_epoch = summary["best_epoch"]
        param_count = summary["parameters"]
        train_time = summary.get("total_time_s", 0) / 60.0
        
        # Get metrics at best epoch from history if possible
        if step["run"] in history_dfs:
            hdf = history_dfs[step["run"]]
            row = hdf[hdf["epoch"] == best_epoch]
            if not row.empty:
                train_acc = row.iloc[0]["train_acc"]
                val_acc = row.iloc[0]["val_acc"]
            else:
                train_acc = 0.0
                val_acc = 0.0
        else:
            train_acc = 0.0
            val_acc = summary["val_metrics"]["accuracy"]
            
        test_acc = summary["test_metrics"]["accuracy"]
        test_f1 = summary["test_metrics"]["macro_f1"]
        test_top3 = summary["test_metrics"].get("top3_accuracy", 0.0)
        
        acc_change = (test_acc - prev_test_acc) * 100 if prev_test_acc > 0 else 0.0
        f1_change = (test_f1 - prev_test_f1) * 100 if prev_test_f1 > 0 else 0.0
        
        prev_test_acc = test_acc
        prev_test_f1 = test_f1
        
        rows.append({
            "step": step["run"],
            "run_name": run_name,
            "description": step["desc"],
            "parameters": param_count,
            "best_epoch": best_epoch,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "test_macro_f1": test_f1,
            "test_top3_acc": test_top3,
            "overfit_gap": train_acc - val_acc,
            "train_time_min": train_time,
            "acc_change_pct_pt": acc_change,
            "f1_change_pct_pt": f1_change
        })
        
    if missing_runs:
        print(f"Warning: The following runs are missing and will be excluded: {', '.join(missing_runs)}")
        
    if not rows:
        print("No ablation runs found. Exiting.")
        return
        
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "ablation_table.csv", index=False)
    
    # MD table
    md_table = df.to_markdown(index=False, floatfmt=".4f")
    with open(out_dir / "ablation_table.md", "w") as f:
        f.write(md_table)
        
    # JSON for web
    df.to_json(out_dir / "ablation.json", orient="records", indent=4)
    
    # Plot ablation accuracy and f1
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width/2, df["test_acc"], width, label="Test Accuracy", color=purple_500)
    ax.bar(x + width/2, df["test_macro_f1"], width, label="Test Macro F1", color=lavender_200)
    ax.set_xticks(x)
    ax.set_xticklabels([r["description"] for _, r in df.iterrows()], rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Ablation Study Results")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "ablation_accuracy.png", dpi=200)
    plt.close(fig)
    
    # Plot validation curves
    fig, ax = plt.subplots(figsize=(10, 6))
    for run_key in ["custom_v1_plain", "custom_v2_bn", "custom_v3_aug", "custom_v4_weighted", "custom_v5_margnet"]:
        if run_key in history_dfs:
            hdf = history_dfs[run_key]
            ax.plot(hdf["epoch"], hdf["val_acc"], label=run_key)
    ax.set_title("Validation Accuracy Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "ablation_val_curves.png", dpi=200)
    plt.close(fig)
    
    # Plot overfit gap for v1, v2, v3
    fig, ax = plt.subplots(figsize=(10, 6))
    for run_key in ["custom_v1_plain", "custom_v2_bn", "custom_v3_aug"]:
        if run_key in history_dfs:
            hdf = history_dfs[run_key]
            gap = hdf["train_acc"] - hdf["val_acc"]
            ax.plot(hdf["epoch"], gap, label=run_key)
    ax.set_title("Overfitting Gap (Train Acc - Val Acc)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gap")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "ablation_overfit_gap.png", dpi=200)
    plt.close(fig)
    
    # Per-class gain v3 -> v4
    v3_report = Path("experiments") / (prefix + "custom_v3_aug") / "classification_report.csv"
    v4_report = Path("experiments") / (prefix + "custom_v4_weighted") / "classification_report.csv"
    if v3_report.exists() and v4_report.exists():
        df3 = pd.read_csv(v3_report)
        df4 = pd.read_csv(v4_report)
        
        merged = df3.merge(df4, on="class_name", suffixes=("_v3", "_v4"))
        merged["f1_gain"] = merged["f1_v4"] - merged["f1_v3"]
        merged = merged.sort_values("f1_gain")
        
        fig, ax = plt.subplots(figsize=(12, 16))
        colors = [purple_500 if g > 0 else "#C0707F" for g in merged["f1_gain"]]
        bars = ax.barh(merged["class_name"], merged["f1_gain"], color=colors)
        
        # Annotate with training counts. Need stats.json
        stats_path = Path("data/processed/stats.json")
        if stats_path.exists():
            with open(stats_path, "r") as f:
                stats = json.load(f)
            counts = stats.get("class_counts_train", {})
            for i, (idx, row) in enumerate(merged.iterrows()):
                c_name = row["class_name"]
                count = counts.get(c_name, 0)
                ax.text(row["f1_gain"], i, f" n={count}", va='center', fontsize=6)
                
        ax.set_xlabel("F1 Gain (V4 - V3)")
        ax.set_title("Per-Class F1 Gain from Class Weights")
        ax.tick_params(axis='y', labelsize=6)
        plt.tight_layout()
        fig.savefig(out_dir / "ablation_per_class_gain.png", dpi=200)
        plt.close(fig)
        
    # Generate MD summary
    best_step = df.loc[df["test_macro_f1"].idxmax()]
    hurt_steps = df[df["f1_change_pct_pt"] < 0]
    
    summary_text = f"# Ablation Summary\n\n"
    summary_text += f"The step that helped the most was **{best_step['description']}** with a Macro F1 of {best_step['test_macro_f1']:.4f}.\n"
    if not hurt_steps.empty:
        summary_text += f"The following steps decreased the F1 score: {', '.join(hurt_steps['description'].tolist())}. "
        summary_text += "Note that a step which does not improve accuracy is still a valid finding in an ablation study.\n"
    else:
        summary_text += "No steps decreased the performance.\n"
        
    summary_text += "\n## Ablation Table\n\n"
    summary_text += md_table
    
    with open(out_dir / "ablation_summary.md", "w") as f:
        f.write(summary_text)
        
    note_block = f"""===== NOTE THIS FOR PPT =====
Ablation Summary:
{md_table}

Best Step: {best_step['description']} (F1: {best_step['test_macro_f1']:.4f})
===== END NOTE ====="""

    with open(out_dir / "note_block.txt", "w") as f:
        f.write(note_block)
        
    print(note_block)

if __name__ == "__main__":
    main()
