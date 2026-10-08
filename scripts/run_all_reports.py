"""
scripts/run_all_reports.py
--------------------------
Runs all reporting scripts (ablation, compare, export) sequentially.
"""
import subprocess
import sys

def main():
    scripts = [
        "scripts.ablation_report",
        "scripts.compare",
        "scripts.export_for_web"
    ]
    for script in scripts:
        print(f"Running {script}...")
        res = subprocess.run([sys.executable, "-m", script])
        if res.returncode != 0:
            print(f"Error running {script}")
            sys.exit(res.returncode)
    print("All reports generated.")

if __name__ == "__main__":
    main()
