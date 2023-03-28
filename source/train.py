import wandb
from torch import cuda
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, set_seed
# , T5ForConditionalGeneration, TrainingArguments
import numpy as np
from evaluate import evaluate_corpus, evaluate_on_asset
import pandas as pd
from paths import WIKILARGE_DATASET, ASSET_DATASET,  PROCESSED_DATA_DIR, \
    WIKILARGE_PROCESSED, ASSET_PROCESSED,  DATASETS_DIR, get_data_filepath, SIMPLIFICATION_DIR, OUTPUT_DIR # ASSET_TRAIN_DATASET, ASSET_TEST_DATASET,
    
import prepare
import paths
from preprocessor import Preprocessor
from model import  tokenize_train, tokenize_test, T5model, tokenizer, training_args, simplify, encoding_test
import evaluate
from utils import read_lines, yield_lines, read_file

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

# perplexity see: https://huggingface.co/docs/transformers/perplexity

if __name__ == '__main__':
    # wandb.login()  
    # wandb.init(project="dutch_simplification")   #wandb.log({'accuracy': train_acc, 'loss': train_loss})
    # wandb.watch(T5model, log="all")
    
    # #Decide ABOUT DATASETS 
    # dataset= prepare.get_train_data(WIKILARGE_DATASET, 30, 40) 
    # print('train_dataset', dataset)
    # # print('pre mapping', dataset['train'][:2])
    # tokenized_dataset = dataset.map((prepare.preprocess_function_train), batched=True, batch_size=1)
    # # print('post mapping', tokenized_dataset['train'][:2])


    # add control tokens BEFORE TRAINING
    # get features_kwargs
    features = {
    # 'WordRatioFeature': {'target_ratio': 0.8},
    'CharRatioFeature': {'target_ratio': 0.8},
    'LevenshteinRatioFeature': {'target_ratio': 0.8},
    'WordRankRatioFeature': {'target_ratio': 0.8},
    'DependencyTreeDepthRatioFeature': {'target_ratio': 0.8}
    }
    # wandb.log({"features":features})
    ## PREPROCESS TRAIN AND VALIDATION DATA
    # preprocessor = Preprocessor(features) # maybe needs to get out of args dict
    # preprocessor.preprocess_dataset(WIKILARGE_DATASET) # dataset)
    
    ## GET TRAIN AND VALIDATION DATASETS
    # trainset_processed = prepare.get_train_data(WIKILARGE_PROCESSED, 0, 10)  
    # print('this is train 0', trainset_processed['train'][0])
    # valset_processed = prepare.get_validation_data(WIKILARGE_PROCESSED, 0,10)
    # print('val 0', valset_processed['validation'][0])
    
    # ## TOKENIZE
    # tokenized_train_dataset = trainset_processed.map((tokenize_train), batched=True, batch_size=1)
    # print('this is the tokenized train dataset', tokenized_train_dataset)
    # tokenized_val_dataset =  valset_processed.map((tokenize_train), batched=True, batch_size=1)
    # print('this is the tokenized val dataset', tokenized_val_dataset)
    
    # # # TEST THE TOKENIZATION
    # encoding_test(tokenized_train_dataset, 'train')
    # encoding_test(tokenized_val_dataset, 'validation')
  
    # data_collator = DataCollatorForSeq2Seq(tokenizer, model=T5model)
    # trainer = Seq2SeqTrainer(model=T5model,
    #                         args=training_args,
    #                         train_dataset=tokenized_train_dataset['train'],
    #                         eval_dataset=tokenized_val_dataset['validation'],
    #                         data_collator=data_collator,
    #                         tokenizer=tokenizer,
    #                         # compute_metrics=compute_metrics
    #                         )
    # set_seed(training_args.seed)
    # trainer.train()
    # trainer.save_model('./saved_model')
    # trainer.evaluate()
    trained_model =  AutoModelForSeq2SeqLM.from_pretrained('./saved_model')
    tokenizer = AutoTokenizer.from_pretrained('./saved_model')
    # # # print(model)
    # print('./saved_model/training_args')
    
    #######NEEDS FIXING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # PREPROCESS TEST DATA (ASSET and WIKILARGE) 
    # preprocessor = Preprocessor(features)
    # preprocessor.preprocess_dataset(ASSET_DATASET)
    # # 2) prepare and tokenize 
    # test_dataset = prepare.get_test_data(ASSET_PROCESSED, 0, 358) # doesnt take first row.
    # print('test_dataset', test_dataset)
    
    # # GENERATION  
    # # takes original data and adds control tokens
    # asset_pfad = get_data_filepath(ASSET_DATASET, 'test', 'orig')
    # predicted_sentences= simplify(asset_pfad, trained_model, tokenizer, features)


    # # EVALUATION & AVERAGES ON SENTENCE LEVEL
    # predictions = read_file(f'{OUTPUT_DIR}/generate/simplification.txt')
    # # # # print('predictions', predictions) 
    # sari_scores, stats = evaluate.calculate_eval_sentence(tokenized_dataset, test_dataset, predictions)
    # # # # EVALUATION & AVERAGES ON CORPUS LEVEL
    # corpus_averages = evaluate.calculate_corpus_averages()
    # # # print(corpus_averages)
    
    
    # add keep delete score 
    # get_corpus_sari_operation_scores(sources, prediction, references)
    # read from file
    
    # # all easse datasets use the 13 a tokenizer - for all languages
    # results = evaluate.evaluate_corpus(features) # give test set here! 

    results_asset = evaluate.evaluate_on_asset(features)    
    
    
    

    

    

    
