
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments
import wandb
from preprocessor import Preprocessor, yield_lines
from paths import OUTPUT_DIR # , # DATASETS_DIR, RESOURCES_DIR, REPO_DIR, WIKILARGE_DATASET
# import prepare

model_checkpoint = "yhavinga/t5-base-dutch" #"yhavinga/t5-v1.1-base-dutch-cased" 
T5model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, use_cache=False) # gradient_checkpointing=True,
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint , additional_special_tokens=None)


# model_name = model_checkpoint.split("/")[-1]
training_args = Seq2SeqTrainingArguments( 
        # f"{wandb.run.name}", 
        num_train_epochs=4,     # trial.suggest_categorical('num_epochs', [2, 3]),
        learning_rate= 1e-4 ,   # trial.suggest_categorical('learning_rate', [1e-4, 1e-3]),  
        per_device_train_batch_size=6,  # trial.suggest_categorical('batch_size', [6, 8, 12, 18]),       
        per_device_eval_batch_size=6,   # trial.suggest_categorical('batch_size', [6, 8, 12, 18]),  
        disable_tqdm=True, 
        predict_with_generate=True,
        gradient_accumulation_steps=4,
        # gradient_checkpointing=True,
        data_seed=12,
        seed = 12, 
        optim="adafactor", 
        adafactor = True, 
        warmup_steps=5,
                
        # evaluation and logging
        evaluation_strategy = "steps", 
        eval_steps = 500, 
        # save_strategy = "steps", 
        save_total_limit=1,
        # logging_strategy="epoch",
        # logging_steps = 500, 
        load_best_model_at_end=True,
        metric_for_best_model = "eval_loss",
        # use_cache=False,
        push_to_hub=False,
        fp16=False,
        remove_unused_columns=True,        
        output_dir="./model_output/", 
        )

def simplify(data, pretrained_model, tokenizer, features_kwargs, output_folder=None):
    max_length = 128
    preprocessor = Preprocessor(features_kwargs)
    for n_line, complex_sent in enumerate(yield_lines(data), start=0):
    # for i in range(0,len(data)):
        print('complex sentence', complex_sent) 
        length = len(complex_sent)
        print(length)
        sentence = preprocessor.encode_sentence(complex_sent)
        print('sentence after preprocessor.encoding', sentence)        
        encoding = tokenizer(sentence, max_length=max_length, return_tensors="pt",add_special_tokens=False)  # ,  padding='max_length')
        input_ids = encoding.input_ids
        output= pretrained_model.generate(inputs = input_ids, 
                # GREEDY DECODING
                # do_sample=False, 
                max_length=50, 
                min_length=3, 
                
                # TOP P, TOP K:
                # do_sample=True,
                # top_k=5,
                # top_p=0.98,
                # top_k=120, 
                # top_p=0.99, 
                
                # BEAM:         
                do_sample=False,
                num_beams=8,
                num_return_sequences=1,
                early_stopping=True,
                repetition_penalty = 1.2, 
                
                # OTHER PARAMETERS:
                # length_penalty= -0.9,
                # no_repeat_ngram_size=4, 
                # suppress_tokens=[4],  
                begin_suppress_tokens= [3,4,7],
                eos_token_id= 4,
                # bad_words_ids = [[0,13,2530,17,4,77]], # List of token ids that are not allowed to be generated. In order to get the token ids of the words that should not appear in the generated text, use tokenizer(bad_words, add_prefix_space=True, add_special_tokens=False).input_ids.
                )
        simplification = tokenizer.decode(output.squeeze(), skip_special_tokens=True, clean_up_tokenization_space=True)
        output_location = f'{OUTPUT_DIR}/generate/simplification.txt' if output_folder is None else output_folder        
        file=open(output_location, "a", encoding="utf8") 
        file.writelines(simplification +'\n')
        file.close()
    return simplification

def tokenize_train(examples):
    max_input_length = 128
    max_target_length = 128
    model_inputs = tokenizer(examples['orig'], max_length=max_input_length, truncation=True, add_special_tokens=False)# , return_tensors='pt', padding=True 
    labels = tokenizer(examples['simp'], max_length=max_target_length, padding=True, truncation=True, add_special_tokens=False) #,  return_tensors='pt', padding=True 
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def tokenize_test(example):
    max_length = 128
    input_ids = tokenizer(example, max_length=max_length, truncation=True,  return_tensors="pt", add_special_tokens=False) #padding='max_length', padding='max_length'
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
    
    # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., 
    # the length of the tokenizer.
def reshape_tokenizer(): # increase the vocabulary model and tokenizer
    # new_tokens = ['-']
    # num_added_toks = tokenizer.add_tokens(new_tokens)
    print('vocab size', T5model.config.vocab_size)
    print('special tokens', tokenizer.additional_special_tokens)