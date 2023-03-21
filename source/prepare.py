from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets, Features, Array2D
import os
from preprocessor import Preprocessor
import pandas as pd
# from model import tokenizer, model
from paths import RESOURCES_DIR, DATASETS_DIR, DUMPS_DIR, OUTPUT_DIR, WIKILARGE_DATASET, ASSET_TEST_DATASET, \
    ASSET_TRAIN_DATASET, WIKILARGE_PROCESSED, ASSET_PROCESSED, PROCESSED_DATA_DIR
from utils import yield_lines, generate_hash

def get_train_data(dataset, begin, end):  
           
    # if dataset == ASSET_TRAIN_DATASET: #  'asset_train': 
    #     folder_path = f'{DATASETS_DIR}/asset/train' #  "./resources/datasets/asset/train/"
    #     file_path = "asset.valid."
    # if dataset == WIKILARGE_DATASET: # 'wikilarge': 
    #     folder_path = f'{DATASETS_DIR}/wikilarge/'# "./resources/datasets/wikilarge/"
    #     file_path = "wikilarge.train."

    
    # checkback if data is processed!
    if dataset == WIKILARGE_PROCESSED: # 'wikilarge': 
        folder_path = f'{PROCESSED_DATA_DIR}/wikilarge/'# "./resources/datasets/wikilarge/"
        file_path = "wikilarge.train."    
    main_dataframe = pd.DataFrame()
    for f in os.listdir(folder_path):
            # if ('.txt' in f):
            header= f.replace(file_path,"").replace("?","")
            # header= header.replace(".txt","").replace("?","")
            # header= header.replace("''","").replace("?","")
            df = pd.read_csv(f"{folder_path}{f}", encoding = 'utf8',sep="\t",header= 0, names=[header])
            # df= pd.DataFrame([row.split(',')]for row in df)
            df.to_csv(f'{OUTPUT_DIR}/train/out_df', encoding='utf8', index=None) 
            main_dataframe = pd.concat([main_dataframe,df],axis=1)    
    main_dataframe.to_csv(f'{OUTPUT_DIR}/train/out_main_dataframe', encoding='utf8', index=None)     
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
        # 'validation': test_valid['train'],
        # 'test': test_valid['test']
        })   # split rule: https://discuss.huggingface.co/t/how-to-split-main-dataset-into-train-dev-test-as-datasetdict/1090
    # dataset.set_format('pt')
    # dataset.set_format('torch', columns=['orig','simp'])
    # print(dataset)
    # print('first sentences of dataset', dataset['train']['orig'][:5], end="\n")
    # print('first sentences of dataset_ COMPLETE', dataset['train']['orig'],end="\n")
    # print('first sentences of dataset', dataset['train']['simp'][:5],end="\n")
    return dataset

def get_validation_data(dataset, begin, end):  
           
    # if dataset == ASSET_TRAIN_DATASET: #  'asset_train': 
    #     folder_path = f'{DATASETS_DIR}/asset/train' #  "./resources/datasets/asset/train/"
    #     file_path = "asset.valid."
    # if dataset == WIKILARGE_DATASET: # 'wikilarge': 
    #     folder_path = f'{DATASETS_DIR}/wikilarge/'# "./resources/datasets/wikilarge/"
    #     file_path = "wikilarge.train."
    
    # checkback if data is processed!
    if dataset == WIKILARGE_PROCESSED: # 'wikilarge': 
        folder_path = f'{PROCESSED_DATA_DIR}/wikilarge/'# "./resources/datasets/wikilarge/"
        file_path = "wikilarge.valid."    
    main_dataframe = pd.DataFrame()
    for f in os.listdir(folder_path):
            # if ('.txt' in f):
            header= f.replace(file_path,"").replace("?","")
            # header= header.replace(".txt","").replace("?","")
            # header= header.replace("''","").replace("?","")
            df = pd.read_csv(f"{folder_path}{f}", encoding = 'utf8',sep="\t",header= 0, names=[header])
            # df= pd.DataFrame([row.split(',')]for row in df)
            df.to_csv(f'{OUTPUT_DIR}/train/out_df', encoding='utf8', index=None) 
            main_dataframe = pd.concat([main_dataframe,df],axis=1)    
    main_dataframe.to_csv(f'{OUTPUT_DIR}/train/out_main_dataframe', encoding='utf8', index=None)     
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
         'validation': test_valid['train'],
        # 'test': test_valid['test']
        })   # split rule: https://discuss.huggingface.co/t/how-to-split-main-dataset-into-train-dev-test-as-datasetdict/1090
    # dataset.set_format('pt')
    # dataset.set_format('torch', columns=['orig','simp'])
    # print(dataset)
    # print('first sentences of dataset', dataset['train']['orig'][:5], end="\n")
    # print('first sentences of dataset_ COMPLETE', dataset['train']['orig'],end="\n")
    # print('first sentences of dataset', dataset['train']['simp'][:5],end="\n")
    return dataset

def get_test_data(dataset, begin, end):
    main_dataframe = pd.DataFrame()  
    if dataset == ASSET_PROCESSED: # 'wikilarge': 
        folder_path = f'{PROCESSED_DATA_DIR}/asset/'# "./resources/datasets/wikilarge/"
        file_path = "asset.test."     
    # if  dataset== ASSET_TEST_DATASET: #  'asset_test': 
    #     folder_path = f'{DATASETS_DIR}/asset/test/'
        for f in os.listdir(folder_path):
            header= f.replace("asset.test.","").replace("?","")
            # header= header.replace(".txt","").replace("?","")
            # header= header.replace("''","").replace("?","")
            df = pd.read_csv(f"{folder_path}{f}", encoding = 'utf8',sep="\t", names=[header]) # header= 0,
            df.to_csv(f'{OUTPUT_DIR}/test/out_df.txt', encoding='utf8',index=None) 
            main_dataframe = pd.concat([main_dataframe,df],axis=1)
        main_dataframe.to_csv(f'{OUTPUT_DIR}/test/out_main_dataframe.txt', encoding='utf8', index=None)          
    test_dataset = DatasetDict({'test': Dataset.from_pandas(main_dataframe)}) # .with_format("torch")
    test_dataset= test_dataset['test'].select(range(begin, end))
    # test_dataset.set_format('torch') 
    # print(test_dataset)
    return test_dataset 

