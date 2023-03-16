# import wandb
from torch import cuda
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, set_seed
# , T5ForConditionalGeneration, TrainingArguments
from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets, Features, Array2D
import numpy as np
import evaluate
import time
import shutil
import pandas as pd
import glob, os
from utils import get_data_filepath, get_dataset_dir, read_lines
from easse.sari import corpus_sari, compute_ngram_stats, get_corpus_sari_operation_scores
import spacy
from torch.utils.data import DataLoader
import csv
from statistics import mean
import re
import paths

WIKILARGE_DATASET = 'wikilarge'
ASSET_TRAIN_DATASET = 'asset_train' # asset validation set
ASSET_TEST_DATASET = 'asset_test'
# WANDB_LOG_MODEL=True
# WANDB_WATCH=all
WANDB_DISABLED = True

def get_train_data_txt(dataset, begin, end):        
    if dataset == 'asset_train': 
        folder_path = "./resources/datasets/asset/train/"
        file_path = "asset.valid."
    if dataset == 'wikilarge': 
        folder_path = "./resources/datasets/wikilarge/"
        file_path = "wikilarge.train."
    main_dataframe = pd.DataFrame()
    for f in os.listdir(folder_path):
            if ('.txt' in f):
                header= f.replace(file_path,"").replace("?","")
                header= header.replace(".txt","").replace("?","")
                # header= header.replace("''","").replace("?","")
                df = pd.read_csv(f"{folder_path}{f}", encoding = 'utf8',sep="\t",header= 0, names=[header])
                # df= pd.DataFrame([row.split(',')]for row in df)
                df.to_csv('./resources/outputs/train/out_df.txt', encoding='utf8', index=None) 
                main_dataframe = pd.concat([main_dataframe,df],axis=1)    
    main_dataframe.to_csv('./resources/outputs/train/out_main_dataframe.txt', encoding='utf8', index=None)     
    #features = Features({"data": Array2D(shape=(2, 2), dtype='int32')})
    # features = Features({'orig': Array2D(shape=(1,1), dtype='string'), 'simp': Array2D(shape=(1,1), dtype='string')})
    #, features=features
    dataset =  Dataset.from_pandas(main_dataframe).with_format("torch")
    #print('main dataset type:', dataset.format['type'] ) # torch
    # print('main dataset features:', dataset.features) # torch

    dataset= dataset.select(range(begin, end))
    # dataloader = DataLoader(dataset.with_format("torch"), batch_size=4)
    # dataset= dataset.select(range(rows))
    
    # SPLIT: 90% train, 10% test + validation
    train_testvalid = dataset.train_test_split(test_size=0.3)
    # Split the 20% test + valid in half test, half valid
    test_valid = train_testvalid['test'].train_test_split(test_size=0.3)
    dataset = DatasetDict({
        'train': train_testvalid['train'],
        'validation': test_valid['train'],
        'test': test_valid['test']})   # split rule: https://discuss.huggingface.co/t/how-to-split-main-dataset-into-train-dev-test-as-datasetdict/1090
    # dataset.set_format('pt')
    # dataset.set_format('torch', columns=['orig','simp'])
    # print(dataset)
    # print('first sentences of dataset', dataset['train']['orig'][:5], end="\n")
    # print('first sentences of dataset_ COMPLETE', dataset['train']['orig'],end="\n")
    # print('first sentences of dataset', dataset['train']['simp'][:5],end="\n")
    return dataset


def get_test_data_txt(dataset, begin, end):        
    main_dataframe = pd.DataFrame()      
    if  dataset== 'asset_test': 
        folder_path = "./resources/datasets/asset/test/"
        for f in os.listdir(folder_path):
            if ('.txt' in f):
                header= f.replace("asset.test.","").replace("?","")
                header= header.replace(".txt","").replace("?","")
                # header= header.replace("''","").replace("?","")
                df = pd.read_csv(f"{folder_path}{f}", encoding = 'utf8',sep="\t", names=[header]) # header= 0,
                df.to_csv('./resources/outputs/test/out_df.txt', encoding='utf8',index=None) 
                main_dataframe = pd.concat([main_dataframe,df],axis=1)
        main_dataframe.to_csv('./resources/outputs/test/out_main_dataframe.txt', encoding='utf8', index=None)          
    test_dataset = DatasetDict({'test': Dataset.from_pandas(main_dataframe)}) # .with_format("torch")
    test_dataset= test_dataset['test'].select(range(begin, end))
    # test_dataset.set_format('torch') 
    # print(test_dataset)
    return test_dataset               

def get_device():
    return 'cuda' if cuda.is_available() else 'cpu'
class T5SimplificationModel():
    def __init__(self, **kwarg):
        # """ Simplification Pytorch lightning module """
        # super(T5SimplificationModel, self).__init__()
        # self.save_hyperparameters()
        model_checkpoint =  "yhavinga/t5-base-dutch" #"yhavinga/t5-v1.1-base-dutch-cased" #"flax-community/t5-base-dutch"#
    
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)# , use_fast=True
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, gradient_checkpointing=True, use_cache=False, return_tensors="pt")

        # self.model = T5ForConditionalGeneration.from_pretrained(self.hparams.model_name).to(self.hparams.device)
        # self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name, use_fast=True)

        self.total_steps = None
        self.predictions = [] 
        
def preprocess_function_train(examples):
    # https://medium.com/nlplanet/a-full-guide-to-finetuning-t5-for-text2text-and-building-a-demo-with-streamlit-c72009631887
    max_input_length = 128
    max_target_length = 128
    model_inputs = tokenizer(examples['orig'], max_length=max_input_length ,  truncation=True, add_special_tokens=False)# , return_tensors='pt', padding=True) # "max_length") # ,  return_tensors='pt'
    labels = tokenizer(examples['simp'], max_length=max_target_length , truncation=True, add_special_tokens=False) #,  return_tensors='pt', padding=True) # "max_length") , return_tensors='pt'

    # not relevant any more bcs padding was kicked! 
    # important: we need to replace the index of the padding tokens by -100 
    # such that they are not taken into account by the CrossEntropyLoss
    # labels_with_ignore_index = []
    # for labels_example in labels:
    #   labels_example = [label if label != 0 else -100 for label in labels_example]
    #   labels_with_ignore_index.append(labels_example)
    
    #       # important: we need to replace the index of the padding tokens by -100
    #   # such that they are not taken into account by the CrossEntropyLoss
    #   labels_with_ignore_index = []
    #   for labels_example in labels:
    #     labels_example = [label if label != 0 else -100 for label in labels_example]
    #     labels_with_ignore_index.append(labels_example)
    # model_inputs["labels"] = labels_with_ignore_index

    model_inputs["labels"] = labels["input_ids"]
    # tokenized_inputs['labels_attention_mask'] =  tokenized_labels['attention_mask']
    # https://discuss.huggingface.co/t/dictionary-of-two-lists-to-datasets-and-fine-tuning-advices-for-fr-it-translation/20393/6
    return model_inputs

def preprocess_function_test(example):
    # https://medium.com/nlplanet/a-full-guide-to-finetuning-t5-for-text2text-and-building-a-demo-with-streamlit-c72009631887
    max_length = 128
    input_ids = tokenizer(example, max_length=max_length, truncation=True,  return_tensors="pt", add_special_tokens=False) # , padding='max_length', padding='max_length') # , )   #padding=True ,
    print('this is the input ids after preprocessing in test', input_ids)
    return input_ids

# model_name = model_checkpoint.split("/")[-1]
training_args = Seq2SeqTrainingArguments(
        # f"-{model_name}",
        # report_to = 'wandb', 
        learning_rate=0.001,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        predict_with_generate=True, # use model for eval
        num_train_epochs=1,
        gradient_accumulation_steps=4,
        # gradient_checkpointing=True,
        # weight_decay= False
        adafactor = True,
        seed = 20, 
        warmup_steps=5,
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
        output_dir="./model_output/"
    )
def encoding_test(examples):
    # sentence 1
    print("encoding test by train method")
    print("sentence 1")
    test_sent1= preprocess_function_train(examples)  ##issue too short! müsste viel länger sein, weil der Paragraph auch viel länger ist!
    print(test_sent1) 
    print('output type after tokenization  ', type(preprocess_function_train(examples)))
    print('output type after tokenization  ', type(test_sent1["input_ids"]))
    print("input_sentence: ", tokenizer.decode(test_sent1["input_ids"]))
    print("input_sentence: ", tokenizer.convert_ids_to_tokens(test_sent1["input_ids"]))
    print("labels: ", tokenizer.decode(test_sent1["labels"]))
    
def generate(tokenized_test_input, trained_model, tokenizer):
    output = trained_model.generate( 
                tokenized_test_input,  
                do_sample=False, # sampling method makes errors 
                # min_new_tokens=13,
                # max_new_tokens=40, # longer is better!! # max_target_length, #128 # countOfWords as alternative
                # doesnt work?  
                # top_k=0, # either temperature or top_k
                # temperature=0.7,  # more weight to powerful tokens
                # # remove_invalid_values=True
                # num_beams = 8,# preset
                # early_stopping= True,
                # length_penalty= 2.0,
                # top_p=0.9, # top p of probability distribution
                # top_k=2,
                # temperature=0.8,
                # min_length= 30,
                # no_repeat_ngram_size= 3,
                num_beams= 4,
                suppress_tokens=[32003,32004,32005,32006,32007,32008,32009,32010,32011,32012,32013,32014,32015,32016,32017,32018,32019,32020,32021,32022,32023,32024,32025,32026,32027,32028,32029,32030,32031,32032,32033,32034,32035,32036,32037,32038,32039,32040,32041,32042,32043,32044,32045,32046,32047,32048,32049,32050,32051,32052,32053,32054,32055,32056,32057,32058,32059,32060,32061,32062,32063,32064,32065,32066,32067,32068,32069,32070,32071,32072,32073,32074,32075,32076,32077,32078,32079,32080,32081,32082,32083,32084,32085,32086,32087,32088,32089,32090,32091,32092,32093,32094,32095,32096,32097,32098,32099,32100,32101,32102], 
                begin_suppress_tokens= [3,4,7], 
                repetition_penalty=1.3 # CRTL PAPER!
                
                )
    print('This is the output of the generator', output) # output is tensor
    # print(type(output))
    # simplification2 = tokenizer.batch_decode(output.squeeze(), skip_special_tokens=True, clean_up_tokenization_space=True)
    # print('simplification 2  ', simplification2)
    simplification = tokenizer.decode(output.squeeze(), skip_special_tokens=True, clean_up_tokenization_space=True)
    # decode returns a list of strings
    # batc decode returns a 
    file=open("./resources/outputs/generate/simplification.txt", "a", encoding="utf8") 
    file.writelines(simplification)
    file.write("\n")
    file.close()
    # lensimpl = len(simplification.split())
    # print(lensimpl, " words")
    # simplification.replace('. ', '.\n')
    return simplification

def create_simplification_dataset(): 
    folder_path= "./resources/outputs/generate/simplification.txt"
    list = []
    with open(folder_path,  "r", encoding='utf8') as f:
        for line in f:
            line = line.rstrip('\n')
            line = [line]
            list.append(line)
    return list



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


# FROM GRS PAPER
# add, keep, delete = get_corpus_sari_operation_scores(orig_sents=orig_sents, sys_sents=sys_sents,
#                                                      refs_sents=ref_sents)
# overal_sari = (add + keep + delete) / 3
# print(f'overal sari:{overal_sari}\
# add: {add}, keep: {keep}, delete: {delete}')
# return {"overall_sari": overal_sari, "addition": add, "keep": keep, "deletion": delete}

def calculate_eval_sentence(dataset, test_dataset, predictions):
    sari_scores = []
    stats = []
    print('len predictions' , len(predictions)) # 
    print('len orig sentences', len(test_dataset['orig']))
    for i in range(1,len(test_dataset['orig'])):  # range starts at 1 now, source list does not get overwritten
        print(i)
        # SOURCES
        sources = test_dataset['orig'][i].split(",'")  # list with or without orig
        # print('source:', sources)
        # PREDICTIONS
        prediction = predictions[i]
        # REFERENCES
        references = []
        refs = []
        if  'simp.0' in test_dataset.column_names: # 'asset_test':
            for j in range(0, ((test_dataset.num_columns)-1)):
                column_name = 'simp.%d' % (j,)
                ref = test_dataset[column_name][i].split(",'")
                refs.append(ref)
            references = refs
        else: 
            ref = test_dataset['simp'][i].split(",'")
            refs.append(ref)
            references = refs
        c = corpus_sari(sources, prediction, references) # from EASSE!         
        add_score, keep_score, del_score = get_corpus_sari_operation_scores(sources, prediction, references)
        stat = add_score, keep_score, del_score
        print('stat type', type(stat))
        print('stat', stat)
        print('sari', c)
        sari_scores.append(c)
        stats.append(stat)
    with open("resources/outputs/generate/stats.txt", "w", newline='') as f:
            sheet = csv.writer(f)
            sheet.writerow(('add', 'keep', 'del'))
            for stat in stats:
                sheet.writerow(stat)    
        
    f = open("./resources/outputs/generate/sari.txt", "w")
    for c in sari_scores:
        f.write(f"{c}")
        f.write("\n")
    f.close()
    return sari_scores, stats

def calculate_corpus_averages():
    sari_df=  pd.read_csv("./resources/outputs/generate/sari.txt", header=None)
    avg_sari = sari_df.mean().item()
    print('sari_average', avg_sari)
    df = pd.read_csv("./resources/outputs/generate/stats.txt")    
    avg_add= df['add'].mean().item()
    avg_keep = df['keep'].mean().item()
    avg_delete = df['del'].mean().item()
    # print('avgadd', avg_add)
    # print('averages: ', avg_add, avg_keep, avg_delete)
    return avg_sari , avg_add, avg_keep, avg_delete

def reshape_tokenizer(): # increase the vocabulary of Bert model and tokenizer
    # new_tokens = ['-']
    # num_added_toks = tokenizer.add_tokens(new_tokens)
    # extra_ids=0,rint('We have added', num_added_toks, 'tokens')
    # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
    print('vocab size', model.config.vocab_size)
    print('special tokens', tokenizer.additional_special_tokens)
    # print('tokens encoder', tokenizer.added_tokens_encoder)

if __name__ == '__main__':
    # wandb.login()  
    # wandb.init(project="dutch_simplification")   #wandb.log({'accuracy': train_acc, 'loss': train_loss})
    # wandb.watch(model, log="all")
    
    model_checkpoint = "yhavinga/t5-base-dutch" #"yhavinga/t5-v1.1-base-dutch-cased" #"flax-community/t5-base-dutch"#
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint,  use_cache=False) # gradient_checkpointing=True,
    # # REPAIR: model = model.get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)# , extra_ids=0, additional_special_tokens=0)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint , additional_special_tokens=None)# extra_ids=None,
    # # TO DO: get_added_vocab https://huggingface.co/transformers/v4.9.2/main_classes/tokenizer.html
    # # add_special_tokens
    
    # #Decide ABOUT DATASETS 
    dataset= get_train_data_txt(WIKILARGE_DATASET, 30, 40) 
    # print(dataset)
    # print('pre mapping', dataset['train'][:2])
    tokenized_dataset = dataset.map(preprocess_function_train, batched=True, batch_size=1)
    # print('post mapping', tokenized_dataset['train'][:2])

    # # ELSE: 
    # test_dataset = dataset['test'] # is already tokenized
    test_dataset = get_test_data_txt(ASSET_TEST_DATASET, 15, 22)
    print(test_dataset)    
    
    # data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    # trainer = Seq2SeqTrainer(model=model,
    #                         args=training_args,
    #                         train_dataset=tokenized_dataset['train'],
    #                         eval_dataset=tokenized_dataset['validation'],
    #                         data_collator=data_collator,
    #                         tokenizer=tokenizer,
    #                         # compute_metrics=compute_metrics
    #                         )
    # set_seed(training_args.seed)
    # trainer.train()
    # trainer.save_model('./saved_model')
    # trainer.evaluate()
    trained_model=model
    trained_model =  AutoModelForSeq2SeqLM.from_pretrained('./saved_model')
    tokenizer = AutoTokenizer.from_pretrained('./saved_model')
    # # # print(model)
    # print('./saved_model/training_args')
    
    # GENERATION  
    
        # # CUTTRENT TRYout
        # # print(type(test_dataset['orig'])) # list of strings 
        # # EVERYTHING NEEDS TO BE MAPPED!!!
        # tokenized_mapped_test_dataset = test_dataset.map(preprocess_function_test, batched=True, batch_size=1)
        # tokenized_mapped_test_dataset.set_format('torch')
        # print(tokenized_mapped_test_dataset)
        # print('post mapping test', tokenized_mapped_test_dataset['orig'][:3])
        # print('break')
        # # they are not tensors of tensors
        # print('post mapping test inputIDS', tokenized_mapped_test_dataset['input_ids'][:3])
        # generated_dataset = generate(tokenized_mapped_test_dataset, trained_model, tokenizer)
        # print(generated_dataset)
        # print(type(generated_dataset)) # is a list
        # # print("generated_sentences: ", tokenizer.batch_decode())
    
    # Working format
    # for i in range(0,len(test_dataset['orig'])): 
    #     print('test input sentence from dataset[orig]', test_dataset['orig'][i])
    #     tokenized_test_input = preprocess_function_test(test_dataset['orig'][i])
    #     print("tokenized input sentence from test ", tokenized_test_input['input_ids'])
    #     generated_dataset= generate(tokenized_test_input['input_ids'], trained_model, tokenizer)
    #     print('generated data decoded!!: ', generated_dataset)    

    # EVALUATION & AVERAGES ON SENTENCE LEVEL
    # MENTION EASSE
    # assemble all formats, if necessary store 
    predictions = create_simplification_dataset()
    # print('predictions', predictions) 
    sari_scores, stats = calculate_eval_sentence(dataset, test_dataset, predictions)
    
    # EVALUATION & AVERAGES ON CORPUS LEVEL
    corpus_averages = calculate_corpus_averages()
    print(corpus_averages)
    
    
    
    
    
    
    
    
    

    

    

    
