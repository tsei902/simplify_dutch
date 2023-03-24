from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent)) # fix path

from source.utils import log_stdout
from source.evaluate import evaluate_on_asset
from paths import EXP_DIR
import optuna

def evaluate(params):
    features_kwargs = {
        'WordRatioFeature': {'target_ratio': params['WordRatio']},
        'CharRatioFeature': {'target_ratio': params['CharRatio']},
        'LevenshteinRatioFeature': {'target_ratio': params['LevenshteinRatio']},
        'WordRankRatioFeature': {'target_ratio': params['WordRankRatio']},
        'DependencyTreeDepthRatioFeature': {'target_ratio': params['DepthTreeRatio']}
    }
    # print('into evalaution')
    return evaluate_on_asset(features_kwargs) # , 'test') # , 'eval1')
    # evaluate_on_asset(features_kwargs, 'valid')

def objective(trial: optuna.trial.Trial) -> float:
    params = {
        'WordRatio' : trial.suggest_float('WordRatio', 0.20, 1.0, step=0.1), # 0.05),
        'CharRatio' : trial.suggest_float('CharRatio', 0.20, 1.0, step=0.1), # 0.05),
        'LevenshteinRatio' : trial.suggest_float('LevenshteinRatio', 0.20, 1.0, step= 0.1), #0.05),
        'WordRankRatio' : trial.suggest_float('WordRankRatio', 0.20, 1.0, step=0.1), #0.05),
        'DepthTreeRatio' : trial.suggest_float('DepthTreeRatio', 0.20, 1.0, step=0.1), # 0.05),
    }
    return evaluate(params)

if __name__=='__main__':

    # # # pruner: optuna.pruners.BasePruner = (
    # #     optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    # # )
    # tuning_log_dir = EXP_DIR / 'tuning_logs'
    # tuning_log_dir.mkdir(parents=True, exist_ok=True)
    # i = 1
    # tuning_logs = tuning_log_dir / f'tokens_logs_{i}.txt'
    # while(tuning_logs.exists()):
    #     i += 1
    #     tuning_logs = tuning_log_dir / f'tokens_logs_{i}.txt'
    #     with log_stdout(tuning_logs):
            # study = optuna.create_study(study_name='TS_T5_study', direction="minimize", storage='sqlite:///TS_T5_study.db')
    study = optuna.create_study(study_name='Tokens_study', direction="maximize", load_if_exists=True) # storage=f'sqlite:///{tuning_log_dir}/Tokens_study.db', 
    study.optimize(objective, n_trials=5)
    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))