# -- fix path --
from pathlib import Path;
import sys;
sys.path.append(str(Path(__file__).resolve().parent.parent))  # fix path
import os
import wandb
from model import tokenize_train
from prepare import get_train_data, get_validation_data
from utils import get_max_seq_length, log_stdout
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, set_seed
from preprocessor import Preprocessor, yield_lines
from paths import DATASETS_DIR, OUTPUT_DIR, RESOURCES_DIR, REPO_DIR, WIKILARGE_DATASET, WIKILARGE_PROCESSED
import torch
import optuna
import os.path
from os import path

def objective(trial: optuna.Trial):     
      
    training_args = Seq2SeqTrainingArguments( 
        f"-{'yhavinga/t5-base-dutch'}",
        report_to = 'wandb',        
        # output_dir="./model_output/", 
        learning_rate=trial.suggest_loguniform('learning_rate', low=4e-5, high=0.01),        #   ('learning_rate', 1e-6, 1e-3),
        # weight_decay=trial.suggest_loguniform('weight_decay', WD_MIN, WD_CEIL),        
        num_train_epochs = trial.suggest_categorical('num_epochs', [3, 5, 8]),         
        per_device_train_batch_size= trial.suggest_categorical('batch_size', [6, 8, 12, 18, 32]),       
        per_device_eval_batch_size= trial.suggest_categorical('batch_size', [6, 8, 12, 18, 32]),  
        warmup_steps=5,
        disable_tqdm=True, 
        predict_with_generate=True, # use model for eval
    
        # max_steps=1,
        gradient_accumulation_steps=4,
        # gradient_checkpointing=True,
        # weight_decay= False
        adafactor = True,
        seed = 4, 
        # evaluation and logging
        evaluation_strategy = "epoch", # needs to remain epoch otherwise no tracking of training loss!
        save_strategy = "epoch",
        save_total_limit=3,
        logging_strategy="epoch",
        # logging_steps = 1, 
        load_best_model_at_end=True,
        metric_for_best_model = "eval_loss",
        # use_cache=False,
        push_to_hub=False,
        fp16=False, # True, # shorter bits, more efficient # tensorsneed to be a multiple of 8 # only savings with high batch size
        remove_unused_columns=True
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=t5dmodel)
    trainer = Seq2SeqTrainer(model=t5dmodel,
                            args=training_args,
                            train_dataset=tokenized_train_dataset['train'],
                            eval_dataset=tokenized_val_dataset['validation'],
                            data_collator=data_collator,
                            tokenizer=tokenizer,
                            # compute_metrics=compute_metrics
                            )     
    
    result = trainer.train()     
    return result.training_loss

if __name__ == '__main__':
    wandb.login()  
    wandb.init(project="hyperparameter_tuning")   #wandb.log({'accuracy': train_acc, 'loss': train_loss})
    t5dmodel = AutoModelForSeq2SeqLM.from_pretrained("yhavinga/t5-base-dutch",  use_cache=False) 
    tokenizer = AutoTokenizer.from_pretrained("yhavinga/t5-base-dutch", additional_special_tokens=None)
    wandb.watch(t5dmodel, log="all")
    
    # prepare data
    features = {
    # 'WordRatioFeature': {'target_ratio': 0.8},
    'CharRatioFeature': {'target_ratio': 0.8},
    'LevenshteinRatioFeature': {'target_ratio': 0.8},
    'WordRankRatioFeature': {'target_ratio': 0.8},
    'DependencyTreeDepthRatioFeature': {'target_ratio': 0.8}
    }
    
    # get different amounts of data! 
    preprocessor = Preprocessor(features) 
    preprocessor.preprocess_dataset(WIKILARGE_DATASET) 
    trainset_processed = get_train_data(WIKILARGE_PROCESSED, 0, 10)  
    print(trainset_processed)
    valset_processed = get_validation_data(WIKILARGE_PROCESSED, 0,10)
    tokenized_train_dataset = trainset_processed.map((tokenize_train), batched=True, batch_size=1)
    tokenized_val_dataset =  valset_processed.map((tokenize_train), batched=True, batch_size=1)   
    
    print('Triggering Optuna study')
    study = optuna.create_study(study_name='hp-search-dutch_t5_base', direction='minimize', pruner=optuna.pruners.MedianPruner())
    # optuna.pruners.BasePruner = () 
    # optuna.pruners.NopPruner()
    study.optimize(func=objective, n_trials=10)
    print(study.best_value)
    print(study.best_params)
    print(study.best_trial)
    
    print('Finding study best parameters')
    best_lr = float(study.best_params['learning_rate'])
    best_weight_decay = float(study.best_params['weight_decay'])
    best_epoch = int(study.best_params['num_train_epochs']) 
    
    print('Saving the best Optuna tuned model')
    if not path.exists('model'):
        os.mkdir('model')

    model_path = "model/{}".format('yhavinga/t5-base-dutch')
    t5dmodel.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    
    # #https://python-bloggers.com/2022/08/hyperparameter-tuning-a-transformer-with-optuna/

    



# def run_tuning(trial, params):
#     dataset = WIKILARGE_DATASET
#     args_dict = dict(
#         model_name='t5-small',
#         max_seq_length=get_max_seq_length(dataset),
#         learning_rate=params['learning_rate'],
#         weight_decay=0.1,
#         # adam_epsilon=params['adam_epsilon'],  # 1e-8,
#         warmup_steps=5,
#         train_batch_size=params['batch_size'],
#         eval_batch_size=params['batch_size'],
#         num_train_epochs=params['num_epochs'],
#         gradient_accumulation_steps=16,
#         n_gpu=torch.cuda.device_count(),
#         early_stop_callback=False,
#         fp_16=False,  # if you want to enable 16-bit training then install apex and set this to true
#         opt_level='O1',
#         # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
#         max_grad_norm=1.0,  # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
#         seed=12,
#         nb_sanity_val_steps=0,
#         train_sample_size=0.4,  # 0.3 = 30% , 1 = 100%
#         valid_sample_size=0.2, # 0.2 = 20%
#     )

#     features_kwargs = {
#         'WordRatioFeature': {'target_ratio': 0.8},
#         'CharRatioFeature': {'target_ratio': 0.8},
#         'LevenshteinRatioFeature': {'target_ratio': 0.8},
#         'WordRankRatioFeature': {'target_ratio': 0.8},
#         'DependencyTreeDepthRatioFeature': {'target_ratio': 0.8}
#     }
#     args_dict['features_kwargs'] = features_kwargs
#     return run_train_tuning(args_dict, dataset)


# def objective(trial: optuna.trial.Trial) -> float:
#     params = {
#         'learning_rate': trial.suggest_loguniform('learning_rate', 1e-6, 1e-3),
#         # 'dropout': trial.suggest_uniform('dropout', 0.1, 0.7),
#         'batch_size': trial.suggest_categorical('batch_size', [6, 12, 18, 32]),
#         'num_epochs': trial.suggest_categorical('num_epochs', [3, 5, 8]),
#         # "adam_epsilon": trial.suggest_float("adam_epsilon", 1e-10, 1e-6),

#         # "weight_decay": trial.suggest_float("weight_decay", 1e-12, 1e-1, log=True),

#     }
#     return run_tuning(trial, params)


# if __name__ == '__main__':

#     # pruner: optuna.pruners.BasePruner = (
#     #     optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
#     # )
#     tuning_log_dir = EXP_DIR / 'tuning_logs'
#     tuning_log_dir.mkdir(parents=True, exist_ok=True)
#     i = 1
#     tuning_logs = tuning_log_dir / f'logs_{i}.txt'
#     while tuning_logs.exists():
#         i += 1
#         tuning_logs = tuning_log_dir / f'logs_{i}.txt'

#     with log_stdout(tuning_logs):
#         # study = optuna.create_study(study_name='TS_T5_study', direction="minimize", storage='sqlite:///TS_T5_study.db')
#         study = optuna.create_study(study_name='TS_T5_study', direction="minimize",
#                                     storage=f'sqlite:///{tuning_log_dir}/TS_T5_study.db', load_if_exists=True)
#         study.optimize(objective, n_trials=100)

#         print("Number of finished trials: {}".format(len(study.trials)))

#         print("Best trial:")
#         trial = study.best_trial

#         print("  Value: {}".format(trial.value))

#         print("  Params: ")
#         for key, value in trial.params.items():
#             print("    {}: {}".format(key, value))
