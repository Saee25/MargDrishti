import argparse
import os
import zipfile
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Pack files for Colab")
    parser.add_argument("--detection", action="store_true", help="Include YOLO detection data")
    args = parser.parse_args()

    zip_name = "margdrishti_colab.zip"
    includes = ["ml", "scripts", "configs", "requirements.txt"]
    
    if args.detection:
        includes.append("data/processed/yolo")
    else:
        includes.extend(["data/processed", "experiments/custom_v5_margnet", "experiments/resnet50_frozen", "experiments/resnet50_finetune"])
    
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for include in includes:
            path = Path(include)
            if not path.exists():
                print(f"Warning: {path} does not exist.")
                continue
                
            if path.is_file():
                arcname = path.as_posix()
                zipf.write(path, arcname)
            else:
                for root, dirs, files in os.walk(path):
                    for file in files:
                        file_path = Path(root) / file
                        arcname = file_path.as_posix()
                        zipf.write(file_path, arcname)
                        
    size_bytes = os.path.getsize(zip_name)
    size_mb = size_bytes / (1024 * 1024)
    
    print(f"Created {zip_name} ({size_mb:.2f} MB)")
    
    upload_speed_mbps = 10.0 # Estimate 10 Mbps upload
    upload_speed_mBps = upload_speed_mbps / 8.0
    upload_time_s = size_mb / upload_speed_mBps
    
    print(f"Estimated upload time on a typical home connection (~10 Mbps): {upload_time_s/60:.1f} minutes")
    
    if size_mb > 500:
        print("Warning: Zip file is over 500 MB. You might want to ensure crops are resized to save space.")

if __name__ == "__main__":
    main()
