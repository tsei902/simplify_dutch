# model.gradient_checkpointing_enable()
# model = model.to(device)
# model.resize_token_embeddings(len(tokenizer))
# model = AutoModelForSeq2SeqLM.from_pretrained("./content/drive/My Drive/Transformers/t5-base-dutch") # , from_pt=False)
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
from easse.sari import corpus_sari
import spacy
from torch.utils.data import DataLoader
import csv


WIKILARGE_DATASET = 'wikilarge'
ASSET_TRAIN_DATASET = 'asset_train' # asset validation set
ASSET_TEST_DATASET = 'asset_test'
# WANDB_LOG_MODEL=True
# WANDB_WATCH=all
WANDB_DISABLED = True

def get_train_data_txt(dataset, rows):        
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

    dataset= dataset.select(range(rows))
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


def get_test_data_txt(dataset, rows):        
    main_dataframe = pd.DataFrame()      
    if  dataset== 'asset_test': 
        folder_path = "./resources/datasets/asset/test/"
        for f in os.listdir(folder_path):
            if ('.txt' in f):
                header= f.replace("asset.test.","").replace("?","")
                header= header.replace(".txt","").replace("?","")
                # header= header.replace("''","").replace("?","")
                df = pd.read_csv(f"{folder_path}{f}", encoding = 'utf8',sep="\t",header= 0, names=[header])
                df.to_csv('./resources/outputs/test/out_df.txt', encoding='utf8',index=None) 
                main_dataframe = pd.concat([main_dataframe,df],axis=1)
        main_dataframe.to_csv('./resources/outputs/test/out_main_dataframe.txt', encoding='utf8', index=None)          
    test_dataset = DatasetDict({'test': Dataset.from_pandas(main_dataframe)}) # .with_format("torch")
    test_dataset= test_dataset['test'].select(range(rows))
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
    model_inputs = tokenizer(examples['orig'], max_length=max_input_length ,  truncation=True)# , return_tensors='pt', padding=True) # "max_length") # ,  return_tensors='pt'
    labels = tokenizer(examples['simp'], max_length=max_target_length , truncation=True) #,  return_tensors='pt', padding=True) # "max_length") , return_tensors='pt'

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
    input_ids = tokenizer(example, max_length=max_length, truncation=True,  return_tensors="pt") # , padding='max_length', padding='max_length') # , )   #padding=True ,
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
        num_train_epochs=3,
        gradient_accumulation_steps=4,
        # gradient_checkpointing=True,
        # weight_decay= False
        adafactor = True,
        seed = 4, 
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
                max_new_tokens=40, # longer is better!! # max_target_length, #128 # countOfWords as alternative
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

def evaluate_sari(sources, predictions, references): 
    # from EASSE package
    sari_score = corpus_sari(sources, predictions, references)
    return sari_score

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

# def reformat_sentences(text): 
#     nlp = spacy.load('en_core_web_sm')
#     # text = "How are you today? I hope you have a great day"
#     tokens = nlp(text)
#     print(tokens)
#     for sent in tokens.sents:
#         print(sent.string.strip())

#  constraints: class transformers.ConstraintListState


def calculate_sari(dataset, test_dataset, predictions):
    sari_scores = []
    for i in range(0, len(predictions)):  # range starts at 0
        # print(i)
        # SOURCES
        sources = test_dataset['orig'][i].split(
            ",'")  # list with or without orig
        # print('source:', sources)
        # PREDICTIONS
        prediction = predictions[i]
        # print('prediction:', prediction)

        # REFERENCES
        references = []
        refs = []
        if  'simp.0' in test_dataset.column_names: # 'asset_test':
            # print('asset loop')
            for j in range(0, ((test_dataset.num_columns)-1)):
                column_name = 'simp.%d' % (j,)  # 'simp.'+[j]+""
                ref = test_dataset[column_name][i].split(",'")
                # print(type(ref))
                # print(ref)
                refs.append(ref)
                # print(refs)
            references = refs
        else: 
            # print('inwikilarge loop')
            ref = test_dataset['simp'][i].split(",'")
            # print(type(ref))
            # print('this is ref from wikilarge', ref)
            # refs.append(ref)
            # # print('these are aggregated referencesrefs)
            refs.append(ref)
            # print(refs)
            references = refs
            # print('these are aggregated references', references)
        # print('these are aggregated references', references)
        # references = test_dataset['simp.0'][i].split(",'"),test_dataset['simp.1'][i].split(",'"),test_dataset['simp.2'][i].split(",'") ,test_dataset['simp.3'][i].split(",'"), test_dataset['simp.4'][i].split(",'"), test_dataset['simp.5'][i].split(",'")
        # references = [list(reference) for reference in references]
        # print('references:', references)
        c = corpus_sari(sources, prediction, references)
        # print('sari', c)
        sari_scores.append(c)
    # print(sari_scores)
    f = open("./resources/outputs/generate/sari.txt", "w")
    for c in sari_scores:
        f.write(f"{c}")
        f.write(",")
    f.close()

if __name__ == '__main__':
    # wandb.login()  
    # wandb.init(project="dutch_simplification")   #wandb.log({'accuracy': train_acc, 'loss': train_loss})
    # wandb.watch(model, log="all")
    
    model_checkpoint = "yhavinga/t5-base-dutch" #"yhavinga/t5-v1.1-base-dutch-cased" #"flax-community/t5-base-dutch"#
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    # # TO DO: get_added_vocab https://huggingface.co/transformers/v4.9.2/main_classes/tokenizer.html
    # # add_special_tokens
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, gradient_checkpointing=True, use_cache=False)
    # # REPAIR: model = model.get_device()

    # #Decide ABOUT DATASETS 
    dataset= get_train_data_txt(WIKILARGE_DATASET, 30) 
    print(dataset)
    print('pre mapping', dataset['train'][:2])
    tokenized_dataset = dataset.map(preprocess_function_train, batched=True, batch_size=1)
    print('post mapping', tokenized_dataset['train'][:2])

    vocabeln = tokenizer.vocab
    print(type(vocabeln))
    # print(vocabeln)
    
 
    # with open("./resources/outputs/generate/vocab_list.txt", "w")as f: #, encoding='utf8'
    #     writer = csv.writer(f)
    #     writer.writerow(vocabeln)
    #     f.close()
        
    with open("./resources/outputs/generate/vocab_list.txt", "w",  encoding='utf8')as f:
        f.write(str(vocabeln))
        f.write(",")
        f.close()
   
    
    # with open("./resources/outputs/generate/vocab_list.txt", 'w', encoding='utf8') as fp:
    #     fp.write('\n'.join(vocabeln))
    
    # folder_path= "./resources/outputs/generate/simplification.txt"
    # list = []
    # with open(folder_path,  "r", encoding='utf8') as f:
    #     for line in f:
    #         line = line.rstrip('\n')
    #         line = [line]
    #         list.append(line)
  
    
    # # ELSE: 
    # test_dataset = dataset['test'] # is already tokenized
    test_dataset = get_test_data_txt(ASSET_TEST_DATASET, 6)
    print(test_dataset)
    # print('test dataset type:', test_dataset.format['type'] ) # torch
    # print(type(test_dataset)) # arow dataset
    # print(test_dataset['test'][:2])
    
    
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
    # trained_model =  AutoModelForSeq2SeqLM.from_pretrained('./saved_model')
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
    for i in range(0,len(test_dataset['orig'])): 
        print('test input sentence from dataset[orig]', test_dataset['orig'][i])
        tokenized_test_input = preprocess_function_test(test_dataset['orig'][i])
        print("tokenized input sentence from test ", tokenized_test_input['input_ids'])
        generated_dataset= generate(tokenized_test_input['input_ids'], trained_model, tokenizer)
        print('generated data decoded!!: ', generated_dataset)

    predictions = create_simplification_dataset()
    # print('predictions', predictions) 
    print('predictions dataset is of type ', type(predictions))

    # EVALUATION
    # assemble all formats, if necessary store
    # first format into list of strings
    # SARI Check only if datset = ASSSET!  
    # we need a list of strings here! 
    sari = calculate_sari(dataset, test_dataset, predictions)
    print(sari)
    
    

    

    
