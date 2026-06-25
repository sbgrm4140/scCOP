## Directory Structure

```text
scMDC_modular/
├── main.py              # Main entry point for argument parsing, data loading, and model training
├── model/               # Core model components
│   ├── __init__.py
│   ├── cop.py           # Core COP network class
│   ├── layers.py        # Network layer building and activation functions
│   └── loss.py          # Loss function definitions (ZINBLoss, NBLoss, SwappedPrediction, etc.)
├── annotation/          # Cell annotation module
│   ├── __init__.py
│   └── annotator.py     # Cell type annotation using OpenAI API
├── utils/               # Utility modules
│   ├── __init__.py
│   ├── data_utils.py    # Data preprocessing (gene filtering, cell filtering, normalization, etc.)
│   └── metrics.py       # Clustering and evaluation metrics computation
├── config.yaml          # Parameter configuration file (to be provided by user)
└── config_utils.py      # Configuration parsing tool (to be provided by user)
```

## Environment Dependencies

Before running the code, please ensure the following core libraries are installed in your environment:
- `torch`
- `scanpy`
- `anndata`
- `numpy`
- `pandas`
- `h5py`
- `scikit-learn`
- `scipy`
- `openai`

## Preparation

Before running the program, make sure that **`config_utils.py`** and **`config.yaml`** are placed in the `scMDC_modular` directory alongside `main.py`.

These two files are responsible for defining and parsing:
- **Data Paths** (input/output paths, H5 file keys, etc.)
- **Model Hyperparameters** (learning rate, network structure, loss weights, etc.)
- **System Configuration** (e.g., `device` to use)

## Usage Guide

Navigate to the `scMDC_modular` directory, and use the `--mode` parameter provided by `main.py` to select the running mode.

### 1. Run Full Mode (with Swap Loss)

The full mode enables the SwappedPrediction loss for cross-modal contrastive learning.

```bash
python main.py --mode full -c config.yaml
```

### 2. Run Ablation Mode (without Swap Loss)

The ablation mode disables the SwappedPrediction loss and its corresponding projection computations to verify the impact of the Swap mechanism on model performance.

```bash
python main.py --mode no_swap -c config.yaml
```

### 3. Run Cell Annotation Mode

The annotation mode uses the OpenAI API to annotate cell types based on the marker genes identified by the model. It reads the `markers_modality1.csv` file generated from the previous steps and outputs `cell_annotations.json` and `cell_annotations.csv`.

```bash
python main.py --mode annotate -c config.yaml
```

## Output Description

After the model finishes training, if corresponding labels are provided (or cluster predictions are made), the following result files will be generated in the `output_path` specified in `config.yaml`:
- `markers_modality1.csv` & `markers_modality2.csv`
- `markers_scores_modality1.csv` & `markers_scores_modality2.csv`
- `cluster_pred.csv`
- `z.csv` 

If you run the annotation mode, the following additional files will be generated:
- `cell_annotations.json`
- `cell_annotations.csv`
- `cluster_pred_with_labels.csv`

Additionally, the final clustering evaluation metrics (ACC, AMI, NMI, ARI) will be printed in the terminal, facilitating metric collection for external scripts.
