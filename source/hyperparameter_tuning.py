import optuna
import wandb
from optuna.integration.wandb import WeightsAndBiasesCallback
# -- fix path --
from pathlib import Path;
import sys;
sys.path.append(str(Path(__file__).resolve().parent.parent))
import wandb
import time
from model import tokenize_train
from prepare import get_train_data, get_validation_data
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from paths import WIKILARGE_PROCESSED
import optuna
from os import path
import os
os.environ["TOKENIZERS_PARALLELISM"]="False"
os.environ["WANDB_SILENT"]="True"
import transformers
transformers.logging.set_verbosity_error()

wandb_kwargs = {"project": "hyperparameter_tuning"}
wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs, as_multirun=True)

@wandbc.track_in_wandb()
def objective(trial):
    
    training_args = Seq2SeqTrainingArguments( 
        f"{wandb.run.name}", 
        num_train_epochs = trial.suggest_categorical('num_epochs', [3, 5, 8]),
        learning_rate=  trial.suggest_float('learning_rate', low=0.00008, high=0.00001),  # , step=0.0005, log=False),             
        per_device_train_batch_size= trial.suggest_categorical('batch_size', [6, 8, 12, 18]),       
        per_device_eval_batch_size= trial.suggest_categorical('batch_size', [6, 8, 12, 18]),  
        optim="adamw_hf",  
        adam_epsilon= trial.suggest_float("adam_epsilon", 1e-10, 1e-6),
        disable_tqdm=True, 
        predict_with_generate=True,
        gradient_accumulation_steps=4,
        # gradient_checkpointing=True,
        weight_decay=0.1,
        seed = 12, 
        warmup_steps=5,
        
        # evaluation and logging
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        save_total_limit=1,
        logging_strategy="epoch",
        logging_steps = 1, 
        load_best_model_at_end=True,
        metric_for_best_model = "eval_loss",
        # use_cache=False,
        push_to_hub=False,
        fp16=False,
        remove_unused_columns=True
    )
    # optimizer = Adafactor(
    #     eps=(1e-30, 1e-3),
    #     clip_threshold=1.0,
    #     decay_rate=-0.8,
    #     beta1=None,
    #     weight_decay=0.1,
    #     relative_step=False,
    #     scale_parameter=False,
    #     warmup_init=False)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=t5dmodel)
    trainer = Seq2SeqTrainer(model=t5dmodel,
                            args=training_args,
                            train_dataset=tokenized_train_dataset['train'],
                            eval_dataset=tokenized_val_dataset['validation'],
                            data_collator=data_collator,
                            tokenizer=tokenizer,
                            # compute_metrics=compute_metrics, 
                            )       
    trainer.train()
    scores = trainer.evaluate() 
    return scores['eval_loss']

if __name__ == '__main__':
    TOKENIZERS_PARALLELISM=False
    t5dmodel = AutoModelForSeq2SeqLM.from_pretrained("yhavinga/t5-base-dutch",  use_cache=False) 
    tokenizer = AutoTokenizer.from_pretrained("yhavinga/t5-base-dutch", additional_special_tokens=None)
    
    features = {
    'WordRatioFeature': {'target_ratio': 0.8},
    'CharRatioFeature': {'target_ratio': 0.8},
    'LevenshteinRatioFeature': {'target_ratio': 0.8},
    'WordRankRatioFeature': {'target_ratio': 0.8},
    'DependencyTreeDepthRatioFeature': {'target_ratio': 0.8}
    }
    
    trainset_processed = get_train_data(WIKILARGE_PROCESSED, 0, 2000)  
    print(trainset_processed)
    valset_processed = get_validation_data(WIKILARGE_PROCESSED, 0,700)
    print(valset_processed)
    tokenized_train_dataset = trainset_processed.map((tokenize_train), batched=True, batch_size=1)
    tokenized_val_dataset =  valset_processed.map((tokenize_train), batched=True, batch_size=1)   
    print('Triggering Optuna study')
    study = optuna.create_study( direction='minimize', pruner=optuna.pruners.PatientPruner(optuna.pruners.MedianPruner(), patience=1, min_delta=0.01)),
    study.optimize(objective, n_trials=20,callbacks=[wandbc],  gc_after_trial=True)