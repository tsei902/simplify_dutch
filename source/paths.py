import torch
from itertools import product
from pathlib import Path
import time 
import tempfile

# PATHS
REPO_DIR = Path(__file__).resolve().parent.parent
RESOURCES_DIR = REPO_DIR / 'resources'
NOTEBOOKS_DIR = REPO_DIR / 'notebooks'
DATASETS_DIR = RESOURCES_DIR / 'datasets'
DUMPS_DIR = RESOURCES_DIR / "DUMPS"
OUTPUT_DIR = RESOURCES_DIR / "outputs"
EXP_DIR = REPO_DIR / 'experiments'
EXP_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR = RESOURCES_DIR / "processed_data"
SIMPLIFICATION_DIR = OUTPUT_DIR/'generate'/'simplification.txt'

# DATASETS
WIKILARGE_DATASET = 'wikilarge'
ASSET_DATASET = 'asset'
# ASSET_TRAIN_DATASET = 'asset/valid' # asset validation set
# ASSET_TEST_DATASET = 'asset/test'
WIKILARGE_PROCESSED = 'wikilarge_processed'
ASSET_PROCESSED = 'asset_processed'

# Word2VecSpecs
WORD_EMBEDDINGS_NAME =  "conllmodel" # "glove.42B.300d" # "combined_320" # "coosto_model" 


# ENVIRONMENT
# WANDB_LOG_MODEL=True
# WANDB_WATCH=all
# WANDB_DISABLED = True
WANDB_MODE = "offline"
WANDB_SILENT = False
TOKENIZERS_PARALLELISM=False

# DEVICE 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LANGUAGES = ['complex', 'simple']
PHASES = ['train', 'valid'] # , 'test']

def get_data_filepath(dataset, phase, type, i=None):
    suffix = ''
    if i is not None:
        suffix = f'.{i}'
    filename = f'{dataset}.{phase}.{type}{suffix}'
    return DATASETS_DIR / dataset / filename

def get_temp_filepath(create=False):
    temp_filepath = Path(tempfile.mkstemp()[1])
    if not create:
        temp_filepath.unlink()
    return temp_filepath

def get_experiment_dir(create_dir=False):
    dir_name = f'{int(time.time() * 1000000)}'
    path = EXP_DIR / f'exp_{dir_name}'
    if create_dir == True: path.mkdir(parents=True, exist_ok=True)
    return path

def get_tuning_log_dir():
    log_dir = EXP_DIR / 'tuning_logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    # i = 1
    # tuning_logs = tuning_log_dir / f'logs_{i}.txt'
    # while(tuning_logs.exists()):
    #     i += 1
    #     tuning_logs = tuning_log_dir / f'logs_{i}.txt'
    return log_dir
def get_last_experiment_dir():
    return sorted(list(EXP_DIR.glob('exp_*')), reverse=True)[0]


if __name__ == '__main__':
    print("REPO", REPO_DIR)
    print('FILEPATH', str(Path(__file__).resolve().parent.parent))
    # print(get_temp_filepath())