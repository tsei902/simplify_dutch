import torch
from itertools import product
from pathlib import Path

# PATHS
REPO_DIR = Path(__file__).resolve().parent.parent
RESOURCES_DIR = REPO_DIR / 'resources'
DATASETS_DIR = RESOURCES_DIR / 'datasets'
DUMPS_DIR = RESOURCES_DIR / "DUMPS"

# DATASETS
WIKILARGE_DATASET = 'wikilarge'
OUTPUT_DIR = RESOURCES_DIR / "outputs"
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