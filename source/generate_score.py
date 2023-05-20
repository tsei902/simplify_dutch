# generate simplifications based on a pretrained model
# open from saved folder, _adam is model pretrained on adam optimizer, _adaf = adafactor optimizer

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from paths import ASSET_DATASET
import evaluate

DEFAULT_METRICS = ['bleu', 'sari', 'fkgl']

features = {
'CharLengthRatioFeature': {'target_ratio': 0.8},
'WordLengthRatioFeature': {'target_ratio': 0.8},
'LevenshteinRatioFeature': {'target_ratio': 0.8},
'WordRankRatioFeature': {'target_ratio': 0.8},
'DependencyTreeDepthRatioFeature': {'target_ratio': 0.8}
} 
# EVALUATION & AVERAGES ON CORPUS LEVEL
# all easse datasets use the 13 a tokenizer - for all languages   
evaluate.evaluate_on_dataset(features, 'saved_model_adaf_10000', ASSET_DATASET, "model_test")    
    
    
