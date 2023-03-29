# generate simplifications based on a pretrained model
# open from saved folder, _adam is model pretrained on adam optimizer, _adaf = adafactor optimizer

import wandb
from torch import cuda
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, set_seed
# , T5ForConditionalGeneration, TrainingArguments
import numpy as np
from evaluate import evaluate_corpus
import pandas as pd
from paths import WIKILARGE_DATASET, ASSET_DATASET, PROCESSED_DATA_DIR, \
    WIKILARGE_PROCESSED, DATASETS_DIR, get_data_filepath, SIMPLIFICATION_DIR, OUTPUT_DIR # ASSET_TRAIN_DATASET, ASSET_TEST_DATASET, 
    
    
import prepare
import paths
from preprocessor import Preprocessor
from model import  tokenize_train, tokenize_test, T5model, tokenizer, training_args, simplify
import evaluate
from utils import read_lines, yield_lines, read_file


trained_model =  AutoModelForSeq2SeqLM.from_pretrained('./saved_model_adam')
tokenizer = AutoTokenizer.from_pretrained('./saved_model_adam')


features = {
'WordRatioFeature': {'target_ratio': 0.8},
'CharRatioFeature': {'target_ratio': 0.8},
'LevenshteinRatioFeature': {'target_ratio': 0.8},
'WordRankRatioFeature': {'target_ratio': 0.8},
'DependencyTreeDepthRatioFeature': {'target_ratio': 0.8}
} 

# # # simplify method assumes no preprocessed data! (preprocessing is done in simplify method)
asset_pfad = get_data_filepath(ASSET_DATASET, 'test', 'orig')
predicted_sentences= simplify(asset_pfad, trained_model, tokenizer, features)


# # EVALUATION & AVERAGES ON SENTENCE LEVEL
# # MENTION EASSE
# # all easse datasets use the 13 a tokenizer - for all languages
results = evaluate.evaluate_corpus(features) # give test set here! 
 
    
    