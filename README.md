# MargDrishti (मार्गदृष्टि)

A data science lab project comparing a custom CNN built from scratch against a pretrained ResNet50 for Indian traffic sign classification. It includes a local web application to demo the models and present the performance comparison.

## Research Question
How close can a small custom CNN get to a fine-tuned ResNet50 on Indian traffic signs, and at what fraction of the parameters, model size, and inference time?

## Folder Structure
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

## Dataset
Download the "Indian Traffic SignBoards" dataset from Roboflow Universe (version 3 Final - Correct Label, COCO JSON export).
Extract the dataset so that the split folders and annotation files are placed directly under `data/raw/`.

## Setup
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

## Workflow
[Placeholders for future tasks]
