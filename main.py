# model.gradient_checkpointing_enable()
# model = model.to(device)
# model.resize_token_embeddings(len(tokenizer))
# model = AutoModelForSeq2SeqLM.from_pretrained("./content/drive/My Drive/Transformers/t5-base-dutch") # , from_pt=False)
import wandb
from torch import cuda
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, set_seed
# , T5ForConditionalGeneration, TrainingArguments
from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets
import numpy as np
import evaluate
import time
import shutil
import pandas as pd
import glob, os
from utils import get_data_filepath, get_dataset_dir, read_lines

WIKILARGE_DATASET = 'wikilarge'
ASSET_DATASET = "asset"
WANDB_LOG_MODEL=True
WANDB_WATCH=all

# def get_data_csv(dataset): 
#     # where does the data come from? 
#     # aggregated datasets require utf-8 encoding before loading them here, done with notepad++
#     file_dict = "./resources/datasets/asset/ASSET_20 lines_DUTCH.csv"
#     dataset = load_dataset("csv", data_files=file_dict, delimiter= ';') 
#     # dataset = dataset.select_columns('herkomst', 'eenvoudig1')
#     column_names = 'eenvoudig0', 'eenvoudig2', 'eenvoudig3', 'eenvoudig4'
#     dataset = dataset.remove_columns(column_names) 
#     # dataset = dataset.remove_columns('eenvoudig3', 'eenvoudig4')
   
#     # SPLIT: 90% train, 10% test + validation
#     train_testvalid = dataset['train'].train_test_split(test_size=0.3)

#     # Split the 20% test + valid in half test, half valid
#     test_valid = train_testvalid['test'].train_test_split(test_size=0.3)
#     # gather everyone if you want to have a single DatasetDict
#     #print(test_valid)
#     dataset = DatasetDict({
#         'train': train_testvalid['train'],
#         'validation': test_valid['train'],
#         'test': test_valid['test']})
#     print(dataset)
#     return dataset


def get_data_txt(dataset, rows): 
    # where does the data come from? 
    # aggregated datasets require utf-8 encoding before loading them here, done with notepad++
    if dataset == 'asset': 
        # asset_ref_paths = [get_data_filepath('asset', 'test', 'simp', i) for i in range(10)]
        file_dict = "./resources/datasets/asset/train/"
        dataset_original = load_dataset("text", data_dir=file_dict, data_files={"train": "asset.valid.orig.txt"})
        dataset_original = dataset_original.rename_column("text", "original")
        # print(dataset_original)
        dataset_simple = load_dataset("text", data_dir=file_dict, data_files={"train":"asset.valid.simp.2.txt"})
        dataset_simple = dataset_simple.rename_column("text", "simple")
        # print(dataset_simple)
        dataset = concatenate_datasets([dataset_original['train'], dataset_simple['train']], axis=1)
        dataset= dataset.select(range(rows))
        print(dataset)
    if dataset == 'wikilarge': 
        file_dict = "./resources/datasets/wikilarge/"
        dataset_original = load_dataset("text", data_dir=file_dict, data_files={"train": "wikilarge.train.complex9999_dutch.txt"})
        dataset_original = dataset_original.rename_column("text", "original")
        # print(dataset_original)
        dataset_simple = load_dataset("text", data_dir=file_dict, data_files={"train":"wikilarge.train.simple9999_dutch.txt"})
        dataset_simple = dataset_simple.rename_column("text", "simple")
        # print(dataset_simple)
        dataset = concatenate_datasets([dataset_original['train'], dataset_simple['train']], axis=1)
        dataset= dataset.select(range(rows))
        print(dataset)
   
    # SPLIT: 90% train, 10% test + validation
    train_testvalid = dataset.train_test_split(test_size=0.3)
    # Split the 20% test + valid in half test, half valid
    test_valid = train_testvalid['test'].train_test_split(test_size=0.3)
    # gather everyone if you want to have a single DatasetDict
    #print(test_valid)
    dataset = DatasetDict({
        'train': train_testvalid['train'],
        'validation': test_valid['train'],
        'test': test_valid['test']})   # split rule: https://discuss.huggingface.co/t/how-to-split-main-dataset-into-train-dev-test-as-datasetdict/1090
    print(dataset)
    return dataset

def get_test_data_txt(dataset, rows):    
    # UNfinished, needs improvement
    # Permanently changes the pandas settings
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.width', None)
    # pd.set_option('display.max_colwidth', -1)    
    main_dataframe = pd.DataFrame()       
    folder_path = "./resources/datasets/asset/test/"
    for f in os.listdir(folder_path):
        if ('.txt' in f):
            header= f.replace("asset.test.","").replace("?","")
            header= header.replace(".txt","").replace("?","")
            # header= header.replace("''","").replace("?","")
            df = pd.read_csv(f"{folder_path}{f}", encoding = 'utf8',sep="\t",header= 0, names=[header]) # ,  index_col=0, )  #,  error_bad_lines = False)
            df.to_csv('./resources/outputs/test/out_df.csv', encoding='utf8') 
            main_dataframe = pd.concat([main_dataframe,df],axis=1)
    main_dataframe.to_csv('./resources/outputs/test/out_main_dataframe.csv', encoding='utf8')  
    # ALTERNATIVE: store in list and then add
    test_dataset = DatasetDict({'test': Dataset.from_pandas(main_dataframe)})
    test_dataset= test_dataset['test'].select(range(rows))
    return test_dataset               
        
class T5SimplificationModel():
    
    def __init__(self, **kwarg):
        # """ Simplification Pytorch lightning module """
        # super(T5SimplificationModel, self).__init__()
        # self.save_hyperparameters()
        model_checkpoint =   "yhavinga/t5-base-dutch" #"yhavinga/t5-v1.1-base-dutch-cased" #"flax-community/t5-base-dutch"#
    
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, gradient_checkpointing=True, use_cache=False)

        # self.model = T5ForConditionalGeneration.from_pretrained(self.hparams.model_name).to(self.hparams.device)
        # self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name, use_fast=True)

        self.total_steps = None
        self.predictions = []
        
        
def get_device():
    return 'cuda' if cuda.is_available() else 'cpu'

def preprocess_function_train(examples):
    # https://medium.com/nlplanet/a-full-guide-to-finetuning-t5-for-text2text-and-building-a-demo-with-streamlit-c72009631887
    max_input_length = 256
    max_target_length = 256
    model_inputs = tokenizer(examples['original'], max_length=max_input_length,  truncation=True) #  , padding="max_length") # ,  
    # Setup the tokenizer for targets
    #with tokenizer.as_target_tokenizer():
    labels = tokenizer(examples['simple'], max_length=max_target_length, truncation=True)  # , padding="max_length")
    # is padding really needed?
    # Setup the tokenizer for targets
    #with tokenizer.as_target_tokenizer():
     # is padding really needed?

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
    return model_inputs

def preprocess_function_test(example):
    # https://medium.com/nlplanet/a-full-guide-to-finetuning-t5-for-text2text-and-building-a-demo-with-streamlit-c72009631887
    
    max_length = 256
    input_ids = tokenizer(example, max_length=max_length, return_tensors="pt" , truncation=True,  padding=True)
    return input_ids


# model_name = model_checkpoint.split("/")[-1]
training_args = Seq2SeqTrainingArguments(
       #  f"-{model_name}",
        report_to = 'wandb', 
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
def encoding_test(dataset):
    # sentence 1
    print("sentence 1")
    test_sent1= preprocess_function_train(dataset['train'][1])  ##issue too short! müsste viel länger sein, weil der Paragraph auch viel länger ist!
    print(test_sent1) 
    print("input_sentence: ", tokenizer.decode(test_sent1["input_ids"]))
    print("labels: ", tokenizer.decode(test_sent1["labels"]))
    # sentence 2
    print("sentence 2")
    test_sent2 = preprocess_function_train(dataset['train'][2])
    print(test_sent2)
    print("input_sentence: ", tokenizer.decode(test_sent2["input_ids"]))
    print("labels: ", tokenizer.decode(test_sent2["labels"]))
    # sentence 3
    print("sentence 3")
    test_sent3 = preprocess_function_train(dataset['train'][3])
    print(test_sent3)
    print("input_sentence: ", tokenizer.decode(test_sent3["input_ids"]))
    print("labels: ", tokenizer.decode(test_sent3["labels"]))
    # sentence 4
    print("sentence 4")
    test_sent4 = preprocess_function_train(dataset['train'][4])
    print(test_sent4)
    print("input_sentence: ", tokenizer.decode(test_sent4["input_ids"]))
    print("labels: ", tokenizer.decode(test_sent4["labels"]))
    
def generate(tokenized_test_input, trained_model, tokenizer):
     
    output = trained_model.generate( tokenized_test_input,  
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
  
    # try tokenizer.batch_decode(simplification)
    simplification = tokenizer.decode(output.squeeze(), skip_special_tokens=True, clean_up_tokenization_space=True)
    file=open("./resources/outputs/generate/simplification.txt", "a", encoding="utf8") 
    file.writelines(simplification)
    file.write("\n")
    file.close()
    # lensimpl = len(simplification.split())
    # print(lensimpl, " words")
    # simplification.replace('. ', '.\n')
    
    return simplification
    
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


# from ACCESS Code Martin: 
# def get_prediction_on_turkcorpus(simplifier, phase):
#     source_filepath = get_data_filepath('turkcorpus', phase, 'complex')
#     pred_filepath = get_temp_filepath()
#     with mute():
#         simplifier(source_filepath, pred_filepath)
#     return pred_filepath


# def evaluate_simplifier(simplifier, phase):
#     pred_filepath = get_prediction_on_turkcorpus(simplifier, phase)
#     pred_filepath = lowercase_file(pred_filepath)
#     pred_filepath = to_lrb_rrb_file(pred_filepath)
#     return evaluate_system_output(f'turkcorpus_{phase}_legacy',
#                                   sys_sents_path=pred_filepath,
#                                   metrics=['bleu', 'sari_legacy', 'fkgl'],
#                                   quality_estimation=True)


# from evaluate import load
# sari = load("sari")
# from evaluate import load
# sari = load("sari")
# sources=["About 95 species are currently accepted."]
# predictions=["About 95 you now get in."]
# references=[["About 95 species are currently known.","About 95 species are now accepted.","95 species are now accepted."]]
# sari_score = sari.compute(sources=sources, predictions=predictions, references=references)
# https://huggingface.co/spaces/evaluate-metric/sari
    
if __name__ == '__main__':
    # wandb.login()  
    # wandb.init(project="dutch_simplification")   #wandb.log({'accuracy': train_acc, 'loss': train_loss})
    model_checkpoint = "yhavinga/t5-base-dutch" #"yhavinga/t5-v1.1-base-dutch-cased" #"flax-community/t5-base-dutch"#
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, gradient_checkpointing=True, use_cache=False)
    # wandb.watch(model, log="all")
    # dataset= get_data_txt(WIKILARGE_DATASET, 10)  
    # print(dataset['train'][3])  
    # tokenized_datasets = dataset.map(preprocess_function, batched=True)
    # print(tokenized_datasets)
    # time.sleep(7)
    # tests= encoding_test(dataset)
    # data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    # trainer = Seq2SeqTrainer(model=model,args=training_args,train_dataset=tokenized_datasets["train"],
    #     eval_dataset=tokenized_datasets["validation"], # should be validation!!
    #     data_collator=data_collator,
    #     tokenizer=tokenizer,
    #     # compute_metrics=compute_metrics
    #     )
    # set_seed(training_args.seed)
    # trainer.train()
    # trainer.save_model('./saved_model')
    # trainer.evaluate()
    trained_model =  AutoModelForSeq2SeqLM.from_pretrained('./saved_model')
    tokenizer = AutoTokenizer.from_pretrained('./saved_model')
    # print(model)
    print('./saved_model/training_args')
    
    # GENERATION
    test_dataset = get_test_data_txt(ASSET_DATASET, 10)
    # print('orig dutch of test set: ' ,test_dataset['orig.dutch'])
    # test_sent3= test_dataset['orig.dutch'][3]
    # print('test sent 3 :', test_sent3)
    # test_sent3 = preprocess_function_test(test_dataset['orig.dutch'][3])
    # print(test_sent3)
    # test_sent3['input_ids']
    # print("input_sentence: ", tokenizer.decode(test_sent3['input_ids']))
    # # print("labels: ", tokenizer.decode(test_sent3))
    
    # print(type(test_dataset['orig.dutch'])) # list of strings 
    
    # print(type(test_dataset['orig.dutch'][2])) # string
    # tokenized_test_input = preprocess_function_test(test_dataset['orig.dutch'][2])
    
    
    # print('tokenized test input from 2', tokenized_test_input)
    # print('tokenized test input from 2 INPUT IDS: ',tokenized_test_input['input_ids'])
    # print(type(tokenized_test_input['input_ids'])) # is handled as a list and not a tensor! 
    # generated_dataset= generate(tokenized_test_input['input_ids'], trained_model, tokenizer)
    # print(generated_dataset)
    
    for i in range(1,len(test_dataset['orig.dutch'])): 
        
        tokenized_test_input = preprocess_function_test(test_dataset['orig.dutch'][i])
        generated_dataset= generate(tokenized_test_input['input_ids'], trained_model, tokenizer)
        print(generated_dataset)

    
