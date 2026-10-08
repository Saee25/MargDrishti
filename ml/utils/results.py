import csv
from pathlib import Path
from datetime import datetime
from ml.config import get_project_root

def print_note_block(title: str, rows: list[str]):
    """
    Prints a formatted block of text meant for PPT notes.
    """
    print("\n===== NOTE THIS FOR PPT =====")
    print(f"{title}")
    print("-" * len(title))
    for row in rows:
        print(row)
    print("===== END NOTE =====\n")

def append_results_log(run_name: str, title: str, rows: list[str]):
    """
    Appends a formatted note block to reports/RESULTS_LOG.md.
    """
    root = get_project_root()
    log_path = root / "reports" / "RESULTS_LOG.md"
    
    # Create the file with a header if it doesn't exist
    if not log_path.exists():
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("# Project Results Log\n\n")
            f.write("This file tracks key metrics and results across different experiment runs.\n\n")
            
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"## {run_name} - {timestamp}\n")
        f.write(f"### {title}\n")
        f.write("```text\n")
        for row in rows:
            f.write(f"{row}\n")
        f.write("```\n\n")

def upsert_experiment_row(run_name: str, values: dict):
    """
    Inserts or updates a row for the given run_name in reports/experiments.csv.
    """
    root = get_project_root()
    csv_path = root / "reports" / "experiments.csv"
    
    rows = []
    headers = ["run_name"]
    
    # Read existing data
    if csv_path.exists():
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                headers = list(reader.fieldnames)
            for row in reader:
                rows.append(row)
                
    # Update headers with any new keys
    for k in values.keys():
        if k not in headers:
            headers.append(k)
            
    # Check if run_name exists
    updated = False
    for i, row in enumerate(rows):
        if row.get("run_name") == run_name:
            for k, v in values.items():
                rows[i][k] = str(v)
            updated = True
            break
            
    # If not updated, append new row
    if not updated:
        new_row = {"run_name": run_name}
        for k, v in values.items():
            new_row[k] = str(v)
        rows.append(new_row)
        
    # Write back to CSV
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
