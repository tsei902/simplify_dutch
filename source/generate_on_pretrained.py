import wandb
from torch import cuda
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, set_seed
# , T5ForConditionalGeneration, TrainingArguments
import numpy as np
from evaluate import evaluate_corpus, evaluate_on_asset
import pandas as pd
from paths import WIKILARGE_DATASET, ASSET_DATASET, ASSET_TRAIN_DATASET, ASSET_TEST_DATASET, PROCESSED_DATA_DIR, \
    WIKILARGE_PROCESSED, DATASETS_DIR, get_data_filepath, SIMPLIFICATION_DIR, OUTPUT_DIR
    
    
import prepare
import paths
from preprocessor import Preprocessor
from model import  tokenize_train, tokenize_test, T5model, tokenizer, training_args, simplify
import evaluate
from utils import read_lines, yield_lines, read_file

# generate on a pretrained model
# open from saved folder

trained_model =  AutoModelForSeq2SeqLM.from_pretrained('./saved_model')
tokenizer = AutoTokenizer.from_pretrained('./saved_model')
print(trained_model)

features = {
'WordRatioFeature': {'target_ratio': 0.8},
'CharRatioFeature': {'target_ratio': 0.8},
'LevenshteinRatioFeature': {'target_ratio': 0.8},
'WordRankRatioFeature': {'target_ratio': 0.8},
'DependencyTreeDepthRatioFeature': {'target_ratio': 0.8}
} 
# ELSE: 
# # 1) preprocess test data  ASSET
# preprocessor = Preprocessor(features) # maybe needs to get out of args dict
# preprocessor.preprocess_dataset(ASSET_DATASET)
# # 2) prepare and tokenize 
# test_dataset = prepare.get_test_data(ASSET_TEST_DATASET, 0, 358) # doesnt take first row.
# print('test_dataset', test_dataset)

# GENERATION  
# ASSET or WIKILARGE TEST
# here also on asset! 
# preprocess data here first
# asset_pfad =  f'{DATASETS_DIR}/asset' # wikilarge/'f'{ASSET_DATASET}/test/' # 

asset_pfad = get_data_filepath(ASSET_DATASET, 'test', 'orig')

# simplify method assumes no preprocessed data! 
predicted_sentences= simplify(asset_pfad, trained_model, tokenizer, features)


# EVALUATION & AVERAGES ON SENTENCE LEVEL
# MENTION EASSE
# list of lists
predictions = read_file(f'{OUTPUT_DIR}/generate/simplification.txt')


# # # # print('predictions', predictions) 
# sari_scores, stats = evaluate.calculate_eval_sentence(tokenized_dataset, test_dataset, predictions)
# # # # EVALUATION & AVERAGES ON CORPUS LEVEL
# corpus_averages = evaluate.calculate_corpus_averages()
# # # print(corpus_averages)


# # all easse datasets use the 13 a tokenizer - for all languages
# results = evaluate.evaluate_corpus(features) # give test set here! 

# results_asset = evaluate.evaluate_on_asset(features)    
    
    