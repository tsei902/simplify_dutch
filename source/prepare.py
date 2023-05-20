from datasets import  DatasetDict, Dataset # , concatenate_datasets, Features, Array2D, load_dataset,
import os
import pandas as pd
from paths import  OUTPUT_DIR, PROCESSED_DATA_DIR, WIKILARGE_PROCESSED # , WIKILARGE_DATASET, ASSET_PROCESSED, RESOURCES_DIR, DATASETS_DIR, DUMPS_DIR
# from utils import yield_lines, generate_hash
# import wandb
# from preprocessor import Preprocessor


# TRAINING DATA needs preprocessing with preprocessor.py

def get_train_data(dataset, begin, end):  
    if dataset == WIKILARGE_PROCESSED: # 'wikilarge': 
        folder_path = f'{PROCESSED_DATA_DIR}/wikilarge/'# "./resources/datasets/wikilarge/"
        file_path = "wikilarge.train."    
    main_dataframe = pd.DataFrame()
    for f in os.listdir(folder_path):
        if f.startswith("wikilarge.train."):
            header= f.replace(file_path,"").replace("?","")
            # header= header.replace(".txt","").replace("?","")
            # header= header.replace("''","").replace("?","")
            df = pd.read_csv(f"{folder_path}{f}", encoding = 'utf8',sep="\t",names=[header])
            # df= pd.DataFrame([row.split(',')]for row in df)
            df.to_csv(f'{OUTPUT_DIR}/train/out_df', encoding='utf8', index=None) 
            main_dataframe = pd.concat([main_dataframe,df],axis=1)    
    main_dataframe.to_csv(f'{OUTPUT_DIR}/train/out_main_dataframe', encoding='utf8', index=None)     
    dataset =  Dataset.from_pandas(main_dataframe).with_format("torch")
    dataset= dataset.select(range(begin, end))
    dataset = DatasetDict({
        'train': dataset,
        })   
    return dataset

def get_validation_data(dataset, begin, end):  
    if dataset == WIKILARGE_PROCESSED: # 'wikilarge': 
        folder_path = f'{PROCESSED_DATA_DIR}/wikilarge/'# "./resources/datasets/wikilarge/"
        file_path = "wikilarge.valid."    
    main_dataframe = pd.DataFrame()
    for f in os.listdir(folder_path):
        if f.startswith("wikilarge.valid."):
            header= f.replace(file_path,"").replace("?","")
            df = pd.read_csv(f"{folder_path}{f}", encoding = 'utf8',sep="\t", names=[header]) #header= 0,
            df.to_csv(f'{OUTPUT_DIR}/train/out_df', encoding='utf8', index=None) 
            main_dataframe = pd.concat([main_dataframe,df],axis=1)    
    main_dataframe.to_csv(f'{OUTPUT_DIR}/train/out_main_dataframe', encoding='utf8', index=None)     
    dataset =  Dataset.from_pandas(main_dataframe).with_format("torch")
    dataset= dataset.select(range(begin, end))
    dataset = DatasetDict({
        'validation': dataset, 
        })
    return dataset


# def get_test_data(dataset, begin, end):
#     main_dataframe = pd.DataFrame()  
#     if dataset == ASSET_PROCESSED: # 'wikilarge': 
#         folder_path = f'{PROCESSED_DATA_DIR}/asset/'
#         file_path = "asset.test."     
#     # if  dataset== ASSET_TEST_DATASET: #  'asset_test': 
#     #     folder_path = f'{DATASETS_DIR}/asset/test/'
#         for f in os.listdir(folder_path):
#             header= f.replace("asset.test.","").replace("?","")
#             # header= header.replace(".txt","").replace("?","")
#             # header= header.replace("''","").replace("?","")
#             df = pd.read_csv(f"{folder_path}{f}", encoding = 'utf8',sep="\t", names=[header])
#             df.to_csv(f'{OUTPUT_DIR}/test/out_df.txt', encoding='utf8',index=None) 
#             main_dataframe = pd.concat([main_dataframe,df],axis=1)
#         main_dataframe.to_csv(f'{OUTPUT_DIR}/test/out_main_dataframe.txt', encoding='utf8', index=None)          
#     test_dataset = DatasetDict({'test': Dataset.from_pandas(main_dataframe)}) 
#     test_dataset= test_dataset['test'].select(range(begin, end))
#     print(test_dataset['test'][0])
#     return test_dataset 

