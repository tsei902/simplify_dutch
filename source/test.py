from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent)) # fix path
from source.evaluate import evaluate_on_dataset
from paths import EXP_DIR
import optuna
import wandb
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, set_seed
from preprocessor import Preprocessor
from optuna.integration.wandb import WeightsAndBiasesCallback
from paths import ASSET_DATASET, WIKILARGE_DATASET, WIKILARGE_PROCESSED, REPO_DIR, PROCESSED_DATA_DIR
import prepare
from model import  tokenize_train, training_args
import os
import glob

# path = PROCESSED_DATA_DIR/f"{'wikilarge'}"

# print(path) 
files = glob.glob('\resources\processed_data\wikilarge')
# files = glob.glob('/resources/processed_data/wikilarge')
for f in files:
    os.remove(f)