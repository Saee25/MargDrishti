import json

with open('notebooks/colab_training.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The user actually said to "add ResNet50 cells... set RUNS to resnet50_frozen then resnet50_finetune"
# Wait! In Colab, we just run `scripts.train` for these.
new_cells = [
    {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [
            '## 8. Run ResNet50 Stage 1 (Frozen Backbone)\n',
            'This trains only the new classification head while keeping the pretrained ResNet50 features frozen. The first run will download the ResNet50 weights from torchvision (about 100 MB). If it disconnects, just re-run this cell.'
        ]
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [
            '!python -m scripts.train --config configs/experiments/resnet50_frozen.yaml --run-name resnet50_frozen'
        ]
    },
    {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [
            '## 9. Run ResNet50 Stage 2 (Fine-Tuning)\n',
            'This unfreezes all layers and trains the whole model with a very small learning rate. It requires the Stage 1 checkpoint to exist. If it disconnects, just re-run this cell and it will resume from last.pt.'
        ]
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [
            'import os\n',
            'if not os.path.exists("experiments/resnet50_frozen/best.pt"):\n',
            '    print("Error: Stage 1 (frozen) best.pt not found. Please run Stage 1 completely before fine-tuning.")\n',
            'else:\n',
            '    !python -m scripts.train --config configs/experiments/resnet50_finetune.yaml --run-name resnet50_finetune'
        ]
    },
    {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [
            '## 10. Download ResNet50 Results\n',
            'This zips the two ResNet50 folders and downloads them to your laptop. Save this file to the root of your MargDrishti project!'
        ]
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [
            '!zip -r -q /content/drive/MyDrive/MargDrishti_Colab/margdrishti_results_resnet.zip experiments/resnet50_frozen experiments/resnet50_finetune\n',
            'from google.colab import files\n',
            'files.download("/content/drive/MyDrive/MargDrishti_Colab/margdrishti_results_resnet.zip")'
        ]
    }
]

# We need to ensure we don't duplicate cells if run twice
has_resnet = any("ResNet50 Stage 1" in "".join(c.get("source", [])) for c in nb["cells"])

if not has_resnet:
    # Insert before the last cell ('If something goes wrong...')
    nb['cells'] = nb['cells'][:-1] + new_cells + [nb['cells'][-1]]
    with open('notebooks/colab_training.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook updated.")
else:
    print("Notebook already contains ResNet50 cells.")
