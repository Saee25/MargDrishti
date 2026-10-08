import argparse
import sys
import time
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments.")
    parser.add_argument("--force", action="store_true", help="Force re-run even if summary.json exists.")
    parser.add_argument("--only", type=str, help="Comma-separated list of runs to execute, e.g. v3,v5.")
    parser.add_argument("--smoke", action="store_true", help="Run a quick smoke test.")
    args = parser.parse_args()

    all_experiments = [
        "custom_v1_plain",
        "custom_v2_bn",
        "custom_v3_aug",
        "custom_v4_weighted",
        "custom_v5_margnet"
    ]

    runs_to_execute = all_experiments
    if args.only:
        only_tags = [t.strip() for t in args.only.split(",")]
        runs_to_execute = []
        for exp in all_experiments:
            if any(tag in exp for tag in only_tags):
                runs_to_execute.append(exp)

    failures = []
    start_time = time.time()
    completed_runs = 0

    print(f"Starting ablation for {len(runs_to_execute)} runs: {runs_to_execute}")

    for idx, run_name in enumerate(runs_to_execute):
        print(f"\n[{idx+1}/{len(runs_to_execute)}] Starting {run_name}...")
        
        actual_run_name = run_name
        if args.smoke:
            actual_run_name = f"_smoke_{run_name}"
            
        out_dir = Path("experiments") / actual_run_name
        summary_path = out_dir / "summary.json"
        last_pt = out_dir / "last.pt"
        
        cmd = [sys.executable, "-m", "scripts.train", "--config", f"configs/experiments/{run_name}.yaml", "--run-name", run_name]
        
        if args.smoke:
            cmd.append("--smoke")
            
        if summary_path.exists() and not args.force:
            print(f"Skipping {run_name}: summary.json already exists (use --force to overwrite).")
            completed_runs += 1
            continue
            
        if last_pt.exists() and not args.force:
            print(f"Resuming {run_name} from {last_pt}...")
            cmd.extend(["--resume", str(last_pt)])
        elif args.force:
            cmd.append("--force")

        run_start = time.time()
        try:
            subprocess.run(cmd, check=True)
            print(f"Successfully finished {run_name}.")
            completed_runs += 1
        except subprocess.CalledProcessError as e:
            print(f"Run {run_name} failed with exit code {e.returncode}.")
            failures.append(run_name)
            
        run_elapsed = time.time() - run_start
        total_elapsed = time.time() - start_time
        
        if completed_runs > 0:
            avg_time = total_elapsed / completed_runs
            remaining_runs = len(runs_to_execute) - (idx + 1)
            estimated_remaining = remaining_runs * avg_time
            print(f"Elapsed time: {total_elapsed/60:.1f}m. Estimated remaining: {estimated_remaining/60:.1f}m.")

    print("\n" + "="*50)
    print("ABLATION RUN COMPLETE")
    print("="*50)
    if failures:
        print(f"The following runs failed: {', '.join(failures)}")
    else:
        print("All runs finished successfully.")

if __name__ == "__main__":
    main()
