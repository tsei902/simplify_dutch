# import wandb
from torch import cuda
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, set_seed
# , T5ForConditionalGeneration, TrainingArguments
import numpy as np
import evaluate
import pandas as pd
from paths import WIKILARGE_DATASET, ASSET_TRAIN_DATASET, ASSET_TEST_DATASET, PROCESSED_DATA_DIR, \
    WIKILARGE_PROCESSED
import prepare
import paths
from preprocessor import Preprocessor
import model
import evaluate

# def compute_metrics(eval_preds):
#     metric = evaluate.load("accuracy", "loss", "BLEU") # perplexity
#     logits, labels = eval_preds
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)

#gradient accumulation steps inbuilt!
    # def evaluate_simplifier(simplifier, phase):
#     pred_filepath = get_prediction_on_turkcorpus(simplifier, phase)
#     pred_filepath = lowercase_file(pred_filepath)
#     pred_filepath = to_lrb_rrb_file(pred_filepath)
#     return evaluate_system_output(f'turkcorpus_{phase}_legacy',
#                                   sys_sents_path=pred_filepath,
#                                   metrics=['bleu', 'sari_legacy', 'fkgl'],
#                                   quality_estimation=True)
#  constraints: class transformers.ConstraintListState

if __name__ == '__main__':
# #     # wandb.login()  
# #     # wandb.init(project="dutch_simplification")   #wandb.log({'accuracy': train_acc, 'loss': train_loss})
# #     # wandb.watch(model, log="all")
    
# #     # #Decide ABOUT DATASETS 
# #     # dataset= prepare.get_train_data(WIKILARGE_DATASET, 30, 40) 
# #     # print('train_dataset', dataset)
# #     # # print('pre mapping', dataset['train'][:2])
# #     # tokenized_dataset = dataset.map((prepare.preprocess_function_train), batched=True, batch_size=1)
# #     # # print('post mapping', tokenized_dataset['train'][:2])

        
    
    
# #     # add control tokens BEFORE TRAINING
# #     # get features_kwargs
    features_kwargs = {
    # 'WordRatioFeature': {'target_ratio': 0.8},
    'CharRatioFeature': {'target_ratio': 0.8},
    'LevenshteinRatioFeature': {'target_ratio': 0.8},
    'WordRankRatioFeature': {'target_ratio': 0.8},
    'DependencyTreeDepthRatioFeature': {'target_ratio': 0.8}
    }
#     preprocessor = Preprocessor(features_kwargs) # maybe needs to get out of args dict
#     preprocessor.preprocess_dataset(WIKILARGE_DATASET) # dataset)

#     # does not retain the dataset!!
#     dataset_preprocessed = prepare.get_train_data(WIKILARGE_PROCESSED, 1, 50) 
#     tokenized_dataset = dataset_preprocessed.map((prepare.preprocess_function_train), batched=True, batch_size=1)
#     # get dataset
#     #decide on format
    
#     # TEST THE TOKENIZATION
    
#     data_collator = DataCollatorForSeq2Seq(model.tokenizer, model=model.model)
#     trainer = Seq2SeqTrainer(model=model.model,
#                             args=model.training_args,
#                             train_dataset=tokenized_dataset['train'],
#                             eval_dataset=tokenized_dataset['validation'],
#                             data_collator=data_collator,
#                             tokenizer=model.tokenizer,
#                             # compute_metrics=compute_metrics
#                             )
#     set_seed(model.training_args.seed)
#     trainer.train()
#     trainer.save_model('./saved_model')
#     trainer.evaluate()
#     trained_model=model
#     trained_model =  AutoModelForSeq2SeqLM.from_pretrained('./saved_model')
#     tokenizer = AutoTokenizer.from_pretrained('./saved_model')
#     # # print(model)
#     print('./saved_model/training_args')
    
#    # # ELSE: 
#     test_dataset = tokenized_dataset['test'] # is already tokenized
#     # test_dataset = prepare.get_test_data(ASSET_TEST_DATASET, 0, 358) # doesnt take first row.
#     print('test_dataset', test_dataset)
    
#     # # GENERATION  
#     predicted_sentences= model.create_generation(test_dataset['orig'], trained_model, tokenizer)








    # # EVALUATION & AVERAGES ON SENTENCE LEVEL
    # # MENTION EASSE
    # # assemble all formats, if necessary store 
    predictions = model.create_simplification_dataset()
    
    # # # print('predictions', predictions) 
    # sari_scores, stats = evaluate.calculate_eval_sentence(tokenized_dataset, test_dataset, predictions)
    
    # # # EVALUATION & AVERAGES ON CORPUS LEVEL
    # corpus_averages = evaluate.calculate_corpus_averages()
    # # print(corpus_averages)
    
    
    # all easse datasets use the 13 a tokenizer - for all languages
    results = evaluate.evaluate_corpus(features_kwargs) # give test set here! 
    # report = evaluate.report()
    # reports = evaluate.get_all_scores()
    
    
    
    
    

    

    

    
