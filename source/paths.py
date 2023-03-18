from itertools import product
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent

RESOURCES_DIR = REPO_DIR / 'resources'
DATASETS_DIR = RESOURCES_DIR / 'datasets'
WIKILARGE_DATASET = 'wikilarge'
ASSET_TRAIN_DATASET = 'asset_train' # asset validation set
ASSET_TEST_DATASET = 'asset_test'