# generate simplifications based on a pretrained model
# open from saved folder, _adam is model pretrained on adam optimizer, _adaf = adafactor optimizer

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM 
from paths import ASSET_DATASET, get_data_filepath, OUTPUT_DIR 
from model import  tokenizer, simplify

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

# EVALUATION OF A GENERATED DATASET
# Note: the simplify method assumes no preprocessed data! (preprocessing is done in simplify method)
pred_filepath = f'{OUTPUT_DIR}/final_decoder_outputs/greedy_test.txt'
asset_pfad = get_data_filepath(ASSET_DATASET, 'test', 'orig') 
print(asset_pfad)
ref_filepaths = [get_data_filepath(ASSET_DATASET, 'test', 'simp', i) for i in range(10)]

# the decoding method is to be specified in model.py
simplify(asset_pfad, trained_model, tokenizer, features,output_folder=pred_filepath)
    
    