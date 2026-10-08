# MargDrishti (मार्गदृष्टि)

A data science lab project comparing a custom CNN built from scratch against a pretrained ResNet50 for Indian traffic sign classification. It includes a local web application to demo the models and present the performance comparison.

## Research Question
How close can a small custom CNN get to a fine-tuned ResNet50 on Indian traffic signs, and at what fraction of the parameters, model size, and inference time?

## Results Table
<!-- RESULTS_START -->
| Model               | Input Size   | Pretrained   |   Parameters |   Trainable Params |   Size (MB) |   MACs (M) |   Epochs Run |   Best Epoch |   Training Time | Device   |   Test Accuracy |   Top-3 Accuracy |   Macro Precision |   Macro Recall |   Macro F1 |   Weighted F1 |   CPU Latency (ms) |   CPU Img/sec |   Times Fewer Params |   Times Smaller |   Times Faster (Latency) |   Acc Gap (pp) |
|:--------------------|:-------------|:-------------|-------------:|-------------------:|------------:|-----------:|-------------:|-------------:|----------------:|:---------|----------------:|-----------------:|------------------:|---------------:|-----------:|--------------:|-------------------:|--------------:|---------------------:|----------------:|-------------------------:|---------------:|
| MargNet             | 64x64        | No           |      1191463 |            1191463 |      4.5693 |   211.1752 |           40 |           35 |          0.0000 | cuda     |          0.0000 |           0.0000 |            0.0000 |         0.0000 |     0.0000 |        0.0000 |             7.7630 |      360.3658 |              19.8525 |         19.8142 |                  10.0809 |         0.0000 |
| ResNet50 frozen     | 224x224      | Yes          |     23653511 |           23653511 |     90.5366 |  4087.2817 |           20 |            6 |          0.0000 | cuda     |          0.0000 |           0.0000 |            0.0000 |         0.0000 |     0.0000 |        0.0000 |            81.1984 |       13.7597 |               1.0000 |          1.0000 |                   0.9638 |         0.0000 |
| ResNet50 fine-tuned | 224x224      | Yes          |     23653511 |           23653511 |     90.5366 |  4087.2817 |           20 |            5 |          0.0000 | cuda     |          0.0000 |           0.0000 |            0.0000 |         0.0000 |     0.0000 |        0.0000 |            78.2577 |       13.5161 |               1.0000 |          1.0000 |                   1.0000 |         0.0000 |
<!-- RESULTS_END -->

## Screenshots
![Demo Screenshot 1](reports/figures/placeholder1.png)
![Demo Screenshot 2](reports/figures/placeholder2.png)

## Dataset Download and Placement
Download the "Indian Traffic SignBoards" dataset from Roboflow Universe (version 3 Final - Correct Label, COCO JSON export).
Extract the dataset so that the split folders and annotation files are placed directly under `data/raw/`.

## Folder Map
- `configs/`: YAML configuration files.
- `data/`: Dataset splits (raw, processed, and web demo samples).
- `ml/`: Python package containing data pipelines, model definitions, and training logic.
- `scripts/`: Entry point scripts for data processing, training, and evaluation.
- `experiments/`: Saved model checkpoints and run-specific files.
- `reports/`: Evaluation results, figures, and CSV records.
- `backend/`: FastAPI application serving the ML models.
- `frontend/`: React + Vite web application.
- `notebooks/`: Google Colab notebooks for remote GPU training.
- `tests/`: Unit tests for the codebase.

## Full Setup (Empty Machine)
### Windows
1. Open PowerShell in the project root.
2. Create and activate the virtual environment:
   ```powershell
   py -3.12 -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
3. Install PyTorch (CPU only, as no NVIDIA GPU was found):
   ```powershell
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```
4. Install other requirements:
   ```powershell
   pip install -r requirements.txt
   ```
5. Install roboflow:
   ```powershell
   pip install roboflow
   ```
6. Setup Node.js frontend:
   ```powershell
   cd frontend
   npm install
   cd ..
   ```

## Workflow and Commands
1. Check Environment (1 min)
   `python -m scripts.check_env`
2. Prepare Data (1-2 mins)
   `python -m scripts.build_crops`
   `python -m scripts.compute_stats`
   `python -m scripts.preview_data`
3. Optional: Time Probe
   `python -m scripts.time_probe --config configs/experiments/custom_v5_margnet.yaml`
4. Pack for Colab (1 min)
   `python -m scripts.pack_for_colab`
5. Train on Google Colab (~1 hour)
   See `notebooks/COLAB_GUIDE.md` for full instructions.
6. Import Results (1 min)
   `python -m scripts.import_run --zip margdrishti_results_custom.zip`
7. Run all reports (1 min)
   `python -m scripts.run_all_reports`
8. Start App
   Windows: `.\start.ps1`
   Unix: `./start.sh`

## Fairness Rules
- Same crop extraction, padding, and splits.
- Same random seed (42).
- Same evaluation metrics and test set.

## Limitations
- Imbalanced classes (many classes have very few samples).
- Small overall dataset size limiting generalisation.
- No real-time object detection yet.

## Troubleshooting
- **CUDA not found**: Use Google Colab for GPU training.
- **DataLoader workers on Windows**: Ensure `if __name__ == '__main__':` guards are used.
- **Port already in use**: Close existing servers or change ports in `start.ps1` / `main.py`.
- **Missing artifacts**: Rerun `python -m scripts.run_all_reports`.
- **Slow CPU training**: Reduce epochs or patience.
- **Colab disconnects**: Run cells incrementally, save checkpoints frequently.

## Licence and Dataset Credit
- Dataset: Indian Traffic SignBoards from Roboflow Universe (by MAJOR PROJECT). MIT Licence.
