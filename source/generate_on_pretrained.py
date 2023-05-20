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

DEFAULT_METRICS = ['bleu', 'sari', 'fkgl']

trained_model =  AutoModelForSeq2SeqLM.from_pretrained('./saved_model')
tokenizer = AutoTokenizer.from_pretrained('./saved_model')

features = {
'CharLengthRatioFeature': {'target_ratio': 0.7},
'WordLengthRatioFeature': {'target_ratio': 0.6},
'LevenshteinRatioFeature': {'target_ratio': 0.6},
'WordRankRatioFeature': {'target_ratio': 0.55},
'DependencyTreeDepthRatioFeature': {'target_ratio': 0.75}
} 

# # # simplify method assumes no preprocessed data! (preprocessing is done in simplify method)
pred_filepath = f'{OUTPUT_DIR}/final_decoder_outputs/beampk120099repearly_full.txt'
asset_pfad = get_data_filepath(ASSET_DATASET, 'test', 'orig') # './translations/sample_asset_google_translate.txt' # 
print(asset_pfad)
ref_filepaths = [get_data_filepath(ASSET_DATASET, 'test', 'simp', i) for i in range(10)]
simplify(asset_pfad, trained_model, tokenizer, features,output_folder=pred_filepath)



# alternatively open and delete empty rows: 
#     with open('path/to/file') as infile, open('output.txt', 'w') as outfile:
#     for line in infile:
#         if not line.strip(): continue  # skip the empty line
#         outfile.write(line)  # non-empty line. Write it to output


# scores = evaluate.evaluate_system_output(test_set="custom", orig_sents_path=asset_pfad, sys_sents_path=str(pred_filepath), refs_sents_paths= ref_filepaths,  lowercase=True,metrics = DEFAULT_METRICS)
# print("SARI: {:.2f} \t BLEU: {:.2f} \t FKGL: {:.2f} ".format(scores['sari'], scores['bleu'], scores['fkgl'])) # test
# # write lines into output dir 
# print(scores)


# # EVALUATION & AVERAGES ON SENTENCE LEVEL
# # MENTION EASSE
# # all easse datasets use the 13 a tokenizer - for all languages
# results = evaluate.evaluate_corpus(features) # give test set here! 
 
    
    