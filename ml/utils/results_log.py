import csv
import datetime
from pathlib import Path

def append_to_results_log(
    run_name,
    model_name,
    variant,
    description,
    param_count,
    model_size_mb,
    epochs_run,
    best_epoch,
    total_time_s,
    device,
    val_acc,
    val_macro_f1,
    test_acc,
    test_top3_acc,
    test_macro_p,
    test_macro_r,
    test_macro_f1,
    test_weighted_f1,
    note_block
):
    # Append to RESULTS_LOG.md
    log_md_path = Path("reports/RESULTS_LOG.md")
    log_md_path.parent.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_md_path, "a", encoding="utf-8") as f:
        f.write(f"\n### {run_name} ({timestamp})\n")
        f.write(f"```text\n{note_block}\n```\n")
        
    # Upsert to experiments.csv
    csv_path = Path("reports/experiments.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    fields = [
        "run_name", "timestamp", "model_name", "variant", "description",
        "param_count", "model_size_mb", "epochs_run", "best_epoch", "total_time_s", "device",
        "val_acc", "val_macro_f1",
        "test_acc", "test_top3_acc", "test_macro_p", "test_macro_r", "test_macro_f1", "test_weighted_f1"
    ]
    
    row_dict = {
        "run_name": run_name,
        "timestamp": timestamp,
        "model_name": model_name,
        "variant": variant,
        "description": description,
        "param_count": param_count,
        "model_size_mb": model_size_mb,
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "total_time_s": total_time_s,
        "device": device,
        "val_acc": val_acc,
        "val_macro_f1": val_macro_f1,
        "test_acc": test_acc,
        "test_top3_acc": test_top3_acc,
        "test_macro_p": test_macro_p,
        "test_macro_r": test_macro_r,
        "test_macro_f1": test_macro_f1,
        "test_weighted_f1": test_weighted_f1
    }
    
    rows = []
    updated = False
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r["run_name"] == run_name:
                    rows.append(row_dict)
                    updated = True
                else:
                    rows.append(r)
                    
    if not updated:
        rows.append(row_dict)
        
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
