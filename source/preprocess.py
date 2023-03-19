from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets, Features, Array2D
import os
import pandas as pd
from model import tokenizer, model
from paths import RESOURCES_DIR, DATASETS_DIR, DUMPS_DIR, OUTPUT_DIR, WIKILARGE_DATASET, ASSET_TEST_DATASET, ASSET_TRAIN_DATASET
from utils import yield_lines, generate_hash, 



def get_train_data_txt(dataset, begin, end):        
    if dataset == ASSET_TRAIN_DATASET: #  'asset_train': 
        folder_path = f'{DATASETS_DIR}/asset/train' #  "./resources/datasets/asset/train/"
        file_path = "asset.valid."
    if dataset == WIKILARGE_DATASET: # 'wikilarge': 
        folder_path = f'{DATASETS_DIR}/wikilarge/'# "./resources/datasets/wikilarge/"
        file_path = "wikilarge.train."
    main_dataframe = pd.DataFrame()
    for f in os.listdir(folder_path):
            if ('.txt' in f):
                header= f.replace(file_path,"").replace("?","")
                header= header.replace(".txt","").replace("?","")
                # header= header.replace("''","").replace("?","")
                df = pd.read_csv(f"{folder_path}{f}", encoding = 'utf8',sep="\t",header= 0, names=[header])
                # df= pd.DataFrame([row.split(',')]for row in df)
                df.to_csv(f'{OUTPUT_DIR}/train/out_df.txt', encoding='utf8', index=None) 
                main_dataframe = pd.concat([main_dataframe,df],axis=1)    
    main_dataframe.to_csv(f'{OUTPUT_DIR}/train/out_main_dataframe.txt', encoding='utf8', index=None)     
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
    if  dataset== ASSET_TEST_DATASET: #  'asset_test': 
        folder_path = f'{DATASETS_DIR}/asset/test/'
        for f in os.listdir(folder_path):
            if ('.txt' in f):
                header= f.replace("asset.test.","").replace("?","")
                header= header.replace(".txt","").replace("?","")
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
    # print('this is the input ids after preprocessing in test', input_ids)
    return input_ids

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
    
def reshape_tokenizer(): # increase the vocabulary of Bert model and tokenizer
    # new_tokens = ['-']
    # num_added_toks = tokenizer.add_tokens(new_tokens)
    # extra_ids=0,rint('We have added', num_added_toks, 'tokens')
    # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
    print('vocab size', model.model.config.vocab_size)
    print('special tokens', tokenizer.additional_special_tokens)
    # print('tokens encoder', tokenizer.added_tokens_encoder)
    
class Preprocessor:
    def __init__(self, features_kwargs=None):
        super().__init__()

        self.features = self.get_features(features_kwargs)
        if features_kwargs:
            self.hash = generate_hash(str(features_kwargs).encode())
        else:
            self.hash = "no_feature"

    def get_class(self, class_name, *args, **kwargs):
        return globals()[class_name](*args, **kwargs)

    def get_features(self, feature_kwargs):
        features = []
        for feature_name, kwargs in feature_kwargs.items():
            features.append(self.get_class(feature_name, **kwargs))
        return features

    def encode_sentence(self, sentence):
        if self.features:
            line = ''
            for feature in self.features:
                line += feature.encode_sentence(sentence) + ' '
            line += ' ' + sentence
            return line.rstrip()
        else:
            return sentence

    def encode_sentence_pair(self, complex_sentence, simple_sentence):
        # print(complex_sentence)
        if self.features:
            line = ''
            for feature in self.features:
                # startTime = timeit.default_timer()
                # print(feature)
                processed_complex, _ = feature.encode_sentence_pair(complex_sentence, simple_sentence)
                line += processed_complex + ' '
                # print(feature, timeit.default_timer() - startTime)
            line += ' ' + complex_sentence
            return line.rstrip()

        else:
            return complex_sentence

    def decode_sentence(self, encoded_sentence):
        for feature in self.features:
            decoded_sentence = feature.decode_sentence(encoded_sentence)
        return decoded_sentence

    def encode_file(self, input_filepath, output_filepath):
        with open(output_filepath, 'w') as f:
            for line in yield_lines(input_filepath):
                f.write(self.encode_sentence(line) + '\n')

    def decode_file(self, input_filepath, output_filepath):
        with open(output_filepath, 'w') as f:
            for line in yield_lines(input_filepath):
                f.write(self.decode_sentence(line) + '\n')

    def process_encode_sentence_pair(self, sentences):
        print(f"{sentences[2]}/{self.line_count}", sentences[0])  # sentence[0] index
        return (self.encode_sentence_pair(sentences[0], sentences[1]))

    def pool_encode_sentence_pair(self, args):
        # print(f"{processed_line_count}/{self.line_count}")
        complex_sent, simple_sent, queue = args
        queue.put(1)
        return self.encode_sentence_pair(complex_sent, simple_sent)

    @print_execution_time
    def encode_file_pair(self, complex_filepath, simple_filepath):
        # print(f"Preprocessing file: {complex_filepath}")
        processed_complex_sentences = []
        self.line_count = count_line(simple_filepath)

        nb_cores = multiprocessing.cpu_count()
        manager = multiprocessing.Manager()
        queue = manager.Queue()

        pool = Pool(processes=nb_cores)
        args = [(complex_sent, simple_sent, queue) for complex_sent, simple_sent in
                yield_sentence_pair(complex_filepath, simple_filepath)]
        res = pool.map_async(self.pool_encode_sentence_pair, args)
        while not res.ready():
            # remaining = res._number_left * res._chunksize
            size = queue.qsize()
            print(f"Preprocessing: {size} / {self.line_count}")
            time.sleep(0.5)
        encoded_sentences = res.get()
        pool.close()
        pool.join()
        # pool.terminate()
        # i = 0
        # for complex_sentence, simple_sentence in yield_sentence_pair(complex_filepath, simple_filepath):
        # # print(complex_sentence)
        #     processed_complex_sentence = self.encode_sentence_pair(complex_sentence, simple_sentence)
        #     i +=1
        #     print(f"{i}/{self.line_count}", processed_complex_sentence)
        # processed_complex_sentences.append(encoded_complex)

        return encoded_sentences

    def get_preprocessed_filepath(self, dataset, phase, type):
        filename = f'{dataset}.{phase}.{type}'
        return self.preprocessed_data_dir / filename

    def preprocess_dataset(self, dataset):
        # download_requirements()
        self.preprocessed_data_dir = PROCESSED_DATA_DIR / self.hash / dataset
        self.preprocessed_data_dir.mkdir(parents=True, exist_ok=True)
        save_preprocessor(self)
        print(f'Preprocessing dataset: {dataset}')

        for phase in PHASES:
            # for phase in ["valid", "test"]:
            complex_filepath = get_data_filepath(dataset, phase, 'complex')
            simple_filepath = get_data_filepath(dataset, phase, 'simple')

            complex_output_filepath = self.preprocessed_data_dir / complex_filepath.name
            simple_output_filepath = self.preprocessed_data_dir / simple_filepath.name
            if complex_output_filepath.exists() and simple_output_filepath.exists():
                continue

            print(f'Prepocessing files: {complex_filepath.name} {simple_filepath.name}')
            processed_complex_sentences = self.encode_file_pair(complex_filepath, simple_filepath)

            write_lines(processed_complex_sentences, complex_output_filepath)
            shutil.copy(simple_filepath, simple_output_filepath)

        print(f'Preprocessing dataset "{dataset}" is finished.')
        return self.preprocessed_data_dir


if __name__ == '__main__':
    features_kwargs = {
        # 'WordRatioFeature': {'target_ratio': 0.8},
        'CharRatioFeature': {'target_ratio': 0.8},
        'LevenshteinRatioFeature': {'target_ratio': 0.8},
        'WordRankRatioFeature': {'target_ratio': 0.8},
        'DependencyTreeDepthRatioFeature': {'target_ratio': 0.8}
    }
    # features_kwargs = {}
    preprocessor = Preprocessor(features_kwargs)
    preprocessor.preprocess_dataset(WIKILARGE_DATASET)
    # preprocessor.preprocess_dataset(NEWSELA_DATASET)