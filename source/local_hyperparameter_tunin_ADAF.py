import optuna
import wandb
from optuna.integration.wandb import WeightsAndBiasesCallback
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import time
from model import tokenize_train
from prepare import get_train_data, get_validation_data
# from utils import get_max_seq_length, log_stdout
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, set_seed
# from preprocessor import Preprocessor, yield_lines
from transformers.optimization import Adafactor, AdafactorSchedule
from paths import WIKILARGE_PROCESSED
import os
os.environ["TOKENIZERS_PARALLELISM"]="False"
os.environ["WANDB_SILENT"]="True"
import transformers
transformers.logging.set_verbosity_error()

wandb_kwargs = {"project": "hyperparameter_tuning_local_adaf"} 
wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs, as_multirun=True)

@wandbc.track_in_wandb()
def objective(trial):
    
    training_args = Seq2SeqTrainingArguments( 
        f"{wandb.run.name}", 
        num_train_epochs=3, # trial.suggest_categorical('num_epochs', [1, 3, 5, 8]),
        learning_rate=  trial.suggest_float('learning_rate', 1e-5, 1e-3), # learning_rate=  trial.suggest_float('learning_rate', 1e-5, 1e-3),
        per_device_train_batch_size=6, # trial.suggest_categorical('batch_size', [6, 8, 12, 18]),       
        per_device_eval_batch_size=6, # trial.suggest_categorical('batch_size', [6, 8, 12, 18]),  
        disable_tqdm=True, 
        predict_with_generate=True,
        gradient_accumulation_steps=4,
        # gradient_checkpointing=True,
        data_seed=12,
        seed = 12, 
        optim="adafactor", 
        adafactor = True, 
        # optim args
        warmup_steps=5, #steps for linear warmup from 0 to learning_rate, Overrides  warmup_ratio
        # If the target learning rate is p and the warm-up period is n, 
        # then the first batch iteration uses 1*p/n for its learning rate;
        # the second uses 2*p/n, and so on: iteration i uses i*p/n, until we hit the nominal rate at iteration n.
        
        # evaluation and logging
        # evaluation_strategy = "steps",
        # save_strategy = "epoch",
        # save_total_limit=1,
        # logging_strategy="epoch",
        # logging_steps = 1, 
        # load_best_model_at_end=True,
        metric_for_best_model = "eval_loss",
        # use_cache=False,
        push_to_hub=False,
        fp16=False,
        remove_unused_columns=True
        )
    
    #  1 Trial only
    # optimizer = Adafactor(params= None, # t5dmodel.parameters(), 
    #                     scale_parameter=True, 
    #                     relative_step=True, 
    #                     warmup_init=True, 
    #                     lr=None)
    # lr_scheduler = AdafactorSchedule(optimizer) # if on or off, does not change much
    
    #Manual external learning rate
    # lr fixed or search parameter above
     # # Paper: Adafactor: Adaptive Learning Rates with Sublinear Memory Cost
    # # # https://arxiv.org/abs/1804.04235 Note that this optimizer internally adjusts the
    # # learning rate depending on the scale_parameter, relative_step and warmup_init options. 
    # # To use a manual (external) learning rate schedule you should set scale_parameter=False
    # # and relative_step=False.
    
    # # https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.Adafactor

    # Since, based on the HF implementation of Adafactor, 
    # in order to use warmup_init, relative_step must be true, 
    # which in turn means that lr must be None.

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=t5dmodel)
    trainer = Seq2SeqTrainer(model=t5dmodel,
                            args=training_args,
                            train_dataset=tokenized_train_dataset['train'],
                            eval_dataset=tokenized_val_dataset['validation'].select(range(4)),
                            data_collator=data_collator,
                            tokenizer=tokenizer,
                            # compute_metrics=compute_metrics, 
                            # optimizers=(optimizer, lr_scheduler), 
                            )       
    trainer.train()
    scores = trainer.evaluate() 
    return scores['eval_loss']

if __name__ == '__main__':
    
    t5dmodel = AutoModelForSeq2SeqLM.from_pretrained("yhavinga/t5-base-dutch",  use_cache=False) 
    tokenizer = AutoTokenizer.from_pretrained("yhavinga/t5-base-dutch", additional_special_tokens=None)
    print(t5dmodel.parameters())
    features = {
    'WordRatioFeature': {'target_ratio': 0.8},
    'CharRatioFeature': {'target_ratio': 0.8},
    'LevenshteinRatioFeature': {'target_ratio': 0.8},
    'WordRankRatioFeature': {'target_ratio': 0.8},
    'DependencyTreeDepthRatioFeature': {'target_ratio': 0.8}
    }
    
    trainset_processed = get_train_data(WIKILARGE_PROCESSED, 0, 20)  
    print(trainset_processed)
    
    valset_processed = get_validation_data(WIKILARGE_PROCESSED, 0,7)
    print(valset_processed)

    tokenized_train_dataset = trainset_processed.map((tokenize_train), batched=True, batch_size=1)
    tokenized_val_dataset =  valset_processed.map((tokenize_train), batched=True, batch_size=1)   
    print('Triggering Optuna study')
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.PatientPruner(optuna.pruners.MedianPruner(), patience=1)) 
    study.optimize(objective, n_trials=2,callbacks=[wandbc],  gc_after_trial=True)
