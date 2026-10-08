import torch
import csv
from ml.utils.seed import set_seed
from ml.utils.results import upsert_experiment_row, append_results_log
from ml.config import get_project_root

def test_set_seed():
    set_seed(42)
    val1 = torch.rand(1).item()
    set_seed(42)
    val2 = torch.rand(1).item()
    assert val1 == val2

def test_results_csv(tmp_path, monkeypatch):
    # Mock project root to tmp_path
    monkeypatch.setattr("ml.utils.results.get_project_root", lambda: tmp_path)
    
    upsert_experiment_row("run1", {"acc": 0.9})
    upsert_experiment_row("run2", {"acc": 0.8})
    upsert_experiment_row("run1", {"acc": 0.95, "loss": 0.1})
    
    csv_path = tmp_path / "reports" / "experiments.csv"
    assert csv_path.exists()
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    assert len(rows) == 2
    assert rows[0]["run_name"] == "run1"
    assert rows[0]["acc"] == "0.95"
    assert rows[0]["loss"] == "0.1"
    assert rows[1]["run_name"] == "run2"
    assert rows[1]["acc"] == "0.8"

def test_results_log(tmp_path, monkeypatch):
    monkeypatch.setattr("ml.utils.results.get_project_root", lambda: tmp_path)
    append_results_log("run1", "Test Title", ["row1", "row2"])
    
    log_path = tmp_path / "reports" / "RESULTS_LOG.md"
    assert log_path.exists()
    
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    assert "## run1" in content
    assert "### Test Title" in content
    assert "row1\nrow2\n" in content
