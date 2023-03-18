import torch
from itertools import product
from pathlib import Path

# PATHS
REPO_DIR = Path(__file__).resolve().parent.parent
RESOURCES_DIR = REPO_DIR / 'resources'
DATASETS_DIR = RESOURCES_DIR / 'datasets'
DUMPS_DIR = RESOURCES_DIR / "DUMPS"
OUTPUT_DIR = RESOURCES_DIR / "outputs"
EXP_DIR = REPO_DIR / 'experiments'
EXP_DIR.mkdir(parents=True, exist_ok=True)

# DATASETS
WIKILARGE_DATASET = 'wikilarge'
ASSET_TRAIN_DATASET = 'asset_train' # asset validation set
ASSET_TEST_DATASET = 'asset_test'

# Word2VecSpecs
WORD_EMBEDDINGS_NAME = "conllmodel"

# WANDB
# WANDB_LOG_MODEL=True
# WANDB_WATCH=all
WANDB_DISABLED = True

# DEVICE 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"