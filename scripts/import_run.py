import argparse
import zipfile
import json
import shutil
import pandas as pd
from pathlib import Path

def update_experiments_csv(csv_path, run_summary, device_override=None):
    if not csv_path.exists():
        df = pd.DataFrame(columns=[
            "run_name", "model_name", "variant", "description", 
            "param_count", "model_size_mb", "epochs_run", "best_epoch", 
            "total_time_s", "device", "val_acc", "val_macro_f1", 
            "test_acc", "test_top3_acc", "test_macro_p", "test_macro_r", 
            "test_macro_f1", "test_weighted_f1"
        ])
    else:
        df = pd.read_csv(csv_path)

    run_name = run_summary.get("run_name")
    
    device = device_override if device_override else run_summary.get("device")

    row = {
        "run_name": run_name,
        "model_name": run_summary.get("model"),
        "variant": run_summary.get("variant"),
        "description": run_summary.get("description", ""),
        "param_count": run_summary.get("parameters"),
        "model_size_mb": run_summary.get("size_mb"),
        "epochs_run": run_summary.get("epochs_run"),
        "best_epoch": run_summary.get("best_epoch"),
        "total_time_s": run_summary.get("total_time_s"),
        "device": device,
        "val_acc": run_summary.get("val_metrics", {}).get("accuracy"),
        "val_macro_f1": run_summary.get("val_metrics", {}).get("macro_f1"),
        "test_acc": run_summary.get("test_metrics", {}).get("accuracy"),
        "test_top3_acc": run_summary.get("test_metrics", {}).get("top3_accuracy"),
        "test_macro_p": run_summary.get("test_metrics", {}).get("macro_precision"),
        "test_macro_r": run_summary.get("test_metrics", {}).get("macro_recall"),
        "test_macro_f1": run_summary.get("test_metrics", {}).get("macro_f1"),
        "test_weighted_f1": run_summary.get("test_metrics", {}).get("weighted_f1")
    }

    if run_name in df["run_name"].values:
        idx = df.index[df["run_name"] == run_name][0]
        for k, v in row.items():
            df.at[idx, k] = v
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    df.to_csv(csv_path, index=False)

def main():
    parser = argparse.ArgumentParser(description="Import runs from a zip file.")
    parser.add_argument("--zip", required=True, type=str, help="Path to results zip file.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing runs.")
    args = parser.parse_args()

    zip_path = Path(args.zip)
    if not zip_path.exists():
        print(f"Error: {zip_path} does not exist.")
        return

    extracted_dir = zip_path.parent / (zip_path.stem + "_extracted")
    extracted_dir.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        zipf.extractall(extracted_dir)
        
    experiments_dest = Path("experiments")
    experiments_dest.mkdir(parents=True, exist_ok=True)
    
    results_log = Path("reports/RESULTS_LOG.md")
    experiments_csv = Path("reports/experiments.csv")
    
    # The zip might contain the runs directly or inside 'experiments/' folder
    # Let's search for summary.json to find run folders
    
    summary_files = list(extracted_dir.rglob("summary.json"))
    
    if not summary_files:
        print("No summary.json found in the zip. Is it a valid results zip?")
        shutil.rmtree(extracted_dir)
        return

    print(f"Found {len(summary_files)} run(s) in {zip_path.name}")
    
    for summary_path in summary_files:
        run_dir_src = summary_path.parent
        run_name = run_dir_src.name
        
        run_dir_dest = experiments_dest / run_name
        
        if run_dir_dest.exists() and not args.force:
            print(f"Skipping {run_name}: already exists. Use --force to overwrite.")
            continue
            
        if run_dir_dest.exists():
            shutil.rmtree(run_dir_dest)
            
        shutil.copytree(run_dir_src, run_dir_dest)
        print(f"Imported {run_name}.")
        
        # Parse summary
        with open(run_dir_dest / "summary.json", "r") as f:
            summary = json.load(f)
            
        device = summary.get("device", "unknown")
        
        update_experiments_csv(experiments_csv, summary, device_override=device)
        
        note_block_file = run_dir_dest / "note_block.txt"
        if note_block_file.exists():
            with open(note_block_file, "r") as f:
                note_block = f.read()
            
            # Print it
            print(f"\n{note_block}\n")
            
            # Append to RESULTS_LOG.md
            with open(results_log, "a") as f:
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"\n\n### Imported Run: {run_name} at {timestamp}\n")
                f.write(f"```text\n{note_block}\n```\n")
                
    shutil.rmtree(extracted_dir)
    print("Import complete.")

if __name__ == "__main__":
    main()
