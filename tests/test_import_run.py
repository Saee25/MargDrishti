import json
import zipfile
import subprocess
import pandas as pd
from pathlib import Path

def test_import_run(tmp_path):
    # Setup fake zip
    fake_zip_path = tmp_path / "fake_results.zip"
    
    # We create a run folder inside the zip
    run_name = "test_run_123"
    
    # Fake summary.json
    summary = {
        "run_name": run_name,
        "model": "TestModel",
        "variant": "v1",
        "description": "test",
        "parameters": 100,
        "size_mb": 1.0,
        "epochs_run": 2,
        "best_epoch": 1,
        "total_time_s": 60,
        "device": "cuda:0",
        "val_metrics": {"accuracy": 0.9, "macro_f1": 0.8},
        "test_metrics": {"accuracy": 0.85, "top3_accuracy": 0.95, "macro_precision": 0.8, "macro_recall": 0.8, "macro_f1": 0.8, "weighted_f1": 0.85}
    }
    
    note_content = "===== NOTE THIS FOR PPT =====\nTest note\n===== END NOTE ====="
    
    with zipfile.ZipFile(fake_zip_path, 'w') as zf:
        zf.writestr(f"{run_name}/summary.json", json.dumps(summary))
        zf.writestr(f"{run_name}/note_block.txt", note_content)
        
    # We need to run import_run from subprocess but it modifies real workspace
    # So we should monkeypatch or just call the python script from a separate cwd or modify script to accept output dirs.
    # Actually, the user asked to add a test using `tmp_path`. Let's mock out the directories.
    
    import sys
    sys.path.append(str(Path.cwd()))
    from scripts.import_run import update_experiments_csv
    
    csv_path = tmp_path / "experiments.csv"
    update_experiments_csv(csv_path, summary)
    
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert len(df) == 1
    assert df.iloc[0]["run_name"] == run_name
    assert df.iloc[0]["device"] == "cuda:0"
