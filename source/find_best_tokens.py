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
from paths import ASSET_DATASET, WIKILARGE_DATASET, WIKILARGE_PROCESSED, REPO_DIR
import prepare
from model import  tokenize_train, training_args
import os
import glob

files = glob.glob('/resources\processed_data\wikilarge')
for f in files:
    os.remove(f)


# WANDB_MODE = "offline"

wandb_kwargs = {"project": "Name_test"}
wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs, as_multirun=True)

def evaluate(params):
    features_kwargs = {
        'WordRatioFeature': {'target_ratio': params['WordRatio']},
        'CharRatioFeature': {'target_ratio': params['CharRatio']},
        'LevenshteinRatioFeature': {'target_ratio': params['LevenshteinRatio']},
        'WordRankRatioFeature': {'target_ratio': params['WordRankRatio']},
        'DependencyTreeDepthRatioFeature': {'target_ratio': params['DepthTreeRatio']}
    }
    return evaluate_on_dataset(features_kwargs, 'saved_model_adaf_10000', ASSET_DATASET, "Tokens_tuning") # takes test file automatically
    

def evaluate_train(params):
    wandb.init(project="hyperparameter_adaf_10000")
    features_kwargs = {
        'WordRatioFeature': {'target_ratio': params['WordRatio']},
        'CharRatioFeature': {'target_ratio': params['CharRatio']},
        #'LevenshteinRatioFeature': {'target_ratio': params['LevenshteinRatio']},
        # 'WordRankRatioFeature': {'target_ratio': params['WordRankRatio']},
        # 'DependencyTreeDepthRatioFeature': {'target_ratio': params['DepthTreeRatio']}
    }
    preprocessor = Preprocessor(features_kwargs) 
    preprocessor.preprocess_dataset(WIKILARGE_DATASET) 
    trainset_processed = prepare.get_train_data(WIKILARGE_PROCESSED, 0, 9999)  
    valset_processed = prepare.get_validation_data(WIKILARGE_PROCESSED, 0,991)
    tokenized_train_dataset = trainset_processed.map((tokenize_train), batched=True, batch_size=1)
    tokenized_val_dataset =  valset_processed.map((tokenize_train), batched=True, batch_size=1)


    model_dir = REPO_DIR /f"{'saved_model_adaf_10000'}"
    print('model dir', model_dir)
    pretrained_model =  AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=pretrained_model)
    trainer = Seq2SeqTrainer(model=pretrained_model,
                            args=training_args,
                            train_dataset=tokenized_train_dataset['train'],
                            eval_dataset=tokenized_val_dataset['validation'],
                            data_collator=data_collator,
                            tokenizer=tokenizer,
                            # compute_metrics=compute_metrics
                            )
    
    set_seed(training_args.seed)
    trainer.train()
    trainer.save_model('./saved_model')    
    # delete 
    [f.unlink() for f in Path("./resources/processed_data/wikilarge").glob("*") if f.is_file()] 
    
    return evaluate_on_dataset(features_kwargs,'source/saved_model', ASSET_DATASET, "Tokens_train_tune") # takes test file automatically
    

def objective(trial: optuna.trial.Trial) -> float:
    params = {
        'WordRatio' : trial.suggest_float('WordRatio', 0.55, 0.95, step= 0.05),
        'CharRatio' : trial.suggest_float('CharRatio', 0.55, 0.95, step=0.05),
       # 'LevenshteinRatio' : trial.suggest_float('LevenshteinRatio', 0.55, 0.90, step= 0.05), 
       #  'WordRankRatio' : trial.suggest_float('WordRankRatio', 0.60, 0.75, step=0.05), 
       #  'DepthTreeRatio' : trial.suggest_float('DepthTreeRatio', 0.80, 0.95, step=0.05),
    }
    return evaluate_train(params)
    # return evaluate(params)

if __name__=='__main__':

    study = optuna.create_study(direction="maximize", load_if_exists=True)  
    study.optimize(objective, n_trials=10, callbacks=[wandbc],  gc_after_trial=True)
    
    # 
