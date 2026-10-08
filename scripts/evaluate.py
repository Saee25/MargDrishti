import argparse
import sys
from pathlib import Path

from ml.engine.evaluator import evaluate

def main():
    parser = argparse.ArgumentParser(description="Re-run evaluation for an existing run.")
    parser.add_argument("--run-name", type=str, required=True, help="Name of the run directory in experiments/.")
    parser.add_argument("--split", type=str, required=True, choices=["train", "val", "test"], help="Dataset split to evaluate.")
    
    args = parser.parse_args()
    
    out_dir = Path("experiments") / args.run_name
    if not out_dir.exists():
        print(f"Error: Run directory {out_dir} does not exist.")
        sys.exit(1)
        
    best_checkpoint = out_dir / "best.pt"
    if not best_checkpoint.exists():
        best_checkpoint = out_dir / "last.pt"
        if not best_checkpoint.exists():
            print(f"Error: Neither best.pt nor last.pt found in {out_dir}.")
            sys.exit(1)
            
    print(f"Evaluating split '{args.split}' using {best_checkpoint}...")
    metrics, _, _ = evaluate(best_checkpoint, args.split, out_dir)
    
    print("\n--- Evaluation Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
