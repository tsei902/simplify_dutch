
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, set_seed
# , T5ForConditionalGeneration, TrainingArguments
# import prepare
from preprocessor import Preprocessor, yield_lines
from paths import DATASETS_DIR, OUTPUT_DIR, RESOURCES_DIR, REPO_DIR, WIKILARGE_DATASET


model_checkpoint = "yhavinga/t5-base-dutch" #"yhavinga/t5-v1.1-base-dutch-cased" #"flax-community/t5-base-dutch"#
T5model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint,  use_cache=False) # gradient_checkpointing=True,
# # REPAIR: model = model.get_device()
# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)# , extra_ids=0, additional_special_tokens=0)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint , additional_special_tokens=None)# extra_ids=None,
# # TO DO: get_added_vocab https://huggingface.co/transformers/v4.9.2/main_classes/tokenizer.html
# # add_special_tokens


# model_name = model_checkpoint.split("/")[-1]
training_args = Seq2SeqTrainingArguments(
        # f"-{model_name}",
        # report_to = 'wandb', 
        learning_rate=0.001,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        predict_with_generate=True, # use model for eval
        num_train_epochs=1,
        max_steps=1,
        gradient_accumulation_steps=4,
        # gradient_checkpointing=True,
        # weight_decay= False
        adafactor = True,
        seed = 12, 
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
        output_dir="./model_output/", 
        remove_unused_columns=True
    )

class T5SimplificationModel():
    def __init__(self, **kwarg):
        # """ Simplification Pytorch lightning module """
        # super(T5SimplificationModel, self).__init__()
        # self.save_hyperparameters()
        model_checkpoint =  "yhavinga/t5-base-dutch" #"yhavinga/t5-v1.1-base-dutch-cased" #"flax-community/t5-base-dutch"#
    
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)# , use_fast=True
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, gradient_checkpointing=True, use_cache=False, return_tensors="pt")

        # # https://discuss.huggingface.co/t/do-you-train-all-layers-when-fine-tuning-t5/1034/5
        # self.block = nn.ModuleList(
        #     [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        # )
        # self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        # self.dropout = nn.Dropout(config.dropout_rate)
        
        # self.model = T5ForConditionalGeneration.from_pretrained(self.hparams.model_name).to(self.hparams.device)
        # self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name, use_fast=True)

        self.total_steps = None
        self.predictions = [] 



def simplify(data, pretrained_model, tokenizer, features_kwargs):
    max_length = 128
    preprocessor = Preprocessor(features_kwargs)
    for n_line, complex_sent in enumerate(yield_lines(data), start=0):
    # for i in range(0,len(data)):
        print('complex sentence', complex_sent) 
        sentence = preprocessor.encode_sentence(complex_sent)
        print(type(sentence))
        # print('sentence after preprocessor.encoding', sentence)
        # sentence = "simplify: " + sentence
        
        encoding = tokenizer(sentence, max_length=max_length, truncation=True,  return_tensors="pt",add_special_tokens=False)  # ,  padding='max_length')
        # print(' TYPE', type(encoding))
        # print('ENCODING AS A WHOLE', encoding)
        input_ids = encoding.input_ids # .to(device)
        # print('ENCODED IDS', input_ids)
        print(type(input_ids))
        # attention_masks = encoding["attention_mask"] # .to(device)
        #print('test input sentence from dataset[orig]', data[i])
        # tokenized_test_input = prepare.tokenize_test(data[i])
        # print("tokenized input sentence from test ", tokenized_test_input['input_ids'])
        output= pretrained_model.generate(inputs = input_ids,  
                do_sample=False, # sampling method makes errors 
                max_length= 50,
                min_length=13, 
                # min_new_tokens=14, 
                # max_new_tokens=200, # if not set, it evaluates to 20 # longer is better!! # max_target_length, #128 # countOfWords as alternative
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
                # eos_token_id= 1,
                suppress_tokens=[32003,32004,32005,32006,32007,32008,32009,32010,32011,32012,32013,32014,32015,32016,32017,32018,32019,32020,32021,32022,32023,32024,32025,32026,32027,32028,32029,32030,32031,32032,32033,32034,32035,32036,32037,32038,32039,32040,32041,32042,32043,32044,32045,32046,32047,32048,32049,32050,32051,32052,32053,32054,32055,32056,32057,32058,32059,32060,32061,32062,32063,32064,32065,32066,32067,32068,32069,32070,32071,32072,32073,32074,32075,32076,32077,32078,32079,32080,32081,32082,32083,32084,32085,32086,32087,32088,32089,32090,32091,32092,32093,32094,32095,32096,32097,32098,32099,32100,32101,32102], 
                begin_suppress_tokens= [3,4,7],
                 # test: bad_words_ids = [[0,13,2530,17,4,77]], # List of token ids that are not allowed to be generated. In order to get the token ids of the words that should not appear in the generated text, use tokenizer(bad_words, add_prefix_space=True, add_special_tokens=False).input_ids.
                repetition_penalty=1.3, # CRTL PAPER!
                # point as an end token
                # suppress any generation of a control token
                )
        print('This is the output of the generator', output) # output is tensor
        print(type(output))
        simplification = tokenizer.decode(output.squeeze(), skip_special_tokens=True, clean_up_tokenization_space=True)
        print('simplification: ', simplification)
        file=open(f'{OUTPUT_DIR}/generate/simplification.txt', "a", encoding="utf8") 
        file.writelines(simplification)
        file.write("\n")
        file.close()
    return simplification

def tokenize_train(examples):
    # https://medium.com/nlplanet/a-full-guide-to-finetuning-t5-for-text2text-and-building-a-demo-with-streamlit-c72009631887
    max_input_length = 128
    max_target_length = 128
    model_inputs = tokenizer(examples['orig'], max_length=max_input_length ,  truncation=True, add_special_tokens=False)# , return_tensors='pt', padding=True) # "max_length") # ,  return_tensors='pt'
    # print('ENCODED IDS', model_inputs)
    # print(type(model_inputs))
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

def tokenize_test(example):
    # https://medium.com/nlplanet/a-full-guide-to-finetuning-t5-for-text2text-and-building-a-demo-with-streamlit-c72009631887
    max_length = 128
    input_ids = tokenizer(example, max_length=max_length, truncation=True,  return_tensors="pt", add_special_tokens=False) # , padding='max_length', padding='max_length') # , )   #padding=True ,
    # print('this is the input ids after preprocessing in test', input_ids)
    return input_ids

def encoding_test(examples, phase):
    # sentence 1
    print("encoding test by", f'{phase}', "method")
    print("sentence 1")
    test_sent1= examples[phase][0]
    print(test_sent1) 
    print('output type after tokenization  ', type(test_sent1))
    print('output type after tokenization  ', type(test_sent1["input_ids"]))
    print("input_sentence: ", tokenizer.decode(test_sent1["input_ids"]))
    # print("input_sentence: ", tokenizer.convert_ids_to_tokens(test_sent1["input_ids"]))
    print("labels: ", tokenizer.decode(test_sent1["labels"]))
    
def reshape_tokenizer(): # increase the vocabulary of Bert model and tokenizer
    # new_tokens = ['-']
    # num_added_toks = tokenizer.add_tokens(new_tokens)
    # extra_ids=0,rint('We have added', num_added_toks, 'tokens')
    # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
    print('vocab size', T5model.config.vocab_size)
    print('special tokens', tokenizer.additional_special_tokens)
    # print('tokens encoder', tokenizer.added_tokens_encoder)
    

