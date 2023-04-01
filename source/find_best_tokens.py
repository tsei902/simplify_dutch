from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent)) # fix path
from source.evaluate import evaluate_on_dataset
from paths import EXP_DIR
import optuna
import wandb
from optuna.integration.wandb import WeightsAndBiasesCallback
from paths import ASSET_DATASET, WIKILARGE_DATASET
WANDB_MODE = "offline"

wandb_kwargs = {"project": "Tokens_tuning"}
wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs, as_multirun=True)

def evaluate(params):
    features_kwargs = {
        'WordRatioFeature': {'target_ratio': params['WordRatio']},
        # 'CharRatioFeature': {'target_ratio': params['CharRatio']},
        # 'LevenshteinRatioFeature': {'target_ratio': params['LevenshteinRatio']},
        # 'WordRankRatioFeature': {'target_ratio': params['WordRankRatio']},
        # 'DependencyTreeDepthRatioFeature': {'target_ratio': params['DepthTreeRatio']}
    }
    return evaluate_on_dataset(features_kwargs, 'saved_model_adam', ASSET_DATASET) # takes test file automatically
    

def objective(trial: optuna.trial.Trial) -> float:
    params = {
        'WordRatio' : trial.suggest_float('WordRatio', 0.20, 1.0, step= 0.05),
        # 'CharRatio' : trial.suggest_float('CharRatio', 0.20, 1.0, step=0.1), # 0.05),
        # 'LevenshteinRatio' : trial.suggest_float('LevenshteinRatio', 0.20, 1.0, step= 0.1), #0.05),
        # 'WordRankRatio' : trial.suggest_float('WordRankRatio', 0.20, 1.0, step=0.1), #0.05),
        # 'DepthTreeRatio' : trial.suggest_float('DepthTreeRatio', 0.20, 1.0, step=0.1), # 0.05),
    }
    return evaluate(params)

if __name__=='__main__':

    study = optuna.create_study(direction="maximize", load_if_exists=True)  
    study.optimize(objective, n_trials=2, callbacks=[wandbc],  gc_after_trial=True)
