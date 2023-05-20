import wandb
from torch import cuda
import paths
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, set_seed
from paths import WIKILARGE_DATASET, WIKILARGE_PROCESSED
import prepare
from preprocessor import Preprocessor
from model import  tokenize_train, tokenize_test, T5model, tokenizer, training_args, simplify, encoding_test


if __name__ == '__main__':
    # wandb.login()  
    # wandb.init(project="dutch_simplification")

    trained_model =  AutoModelForSeq2SeqLM.from_pretrained('./saved_model')
    tokenizer = AutoTokenizer.from_pretrained('./saved_model')

    features = {
    'CharLengthRatioFeature': {'target_ratio': 0.7},
    'WordLengthRatioFeature': {'target_ratio': 0.6},
    'LevenshteinRatioFeature': {'target_ratio': 0.6},
    'WordRankRatioFeature': {'target_ratio': 0.55},
    'DependencyTreeDepthRatioFeature': {'target_ratio': 0.75}
    }
    # wandb.log({"features":features})
    
    ## PREPROCESS TRAIN AND VALIDATION DATA
    preprocessor = Preprocessor(features)
    preprocessor.preprocess_dataset(WIKILARGE_DATASET)
    
    ## GET TRAIN AND VALIDATION DATASETS
    trainset_processed = prepare.get_train_data(WIKILARGE_PROCESSED, 0, 10)  
    print('this is train 0', trainset_processed['train'][1])
    valset_processed = prepare.get_validation_data(WIKILARGE_PROCESSED, 0,2)
    print('val 0', valset_processed['validation'][1])
    
    # ## TOKENIZE
    tokenized_train_dataset = trainset_processed.map((tokenize_train), batched=True, batch_size=1)
    print('this is the tokenized train dataset', tokenized_train_dataset)
    tokenized_val_dataset =  valset_processed.map((tokenize_train), batched=True, batch_size=1)
    print('this is the tokenized val dataset', tokenized_val_dataset)
    
    # # # TEST THE TOKENIZATION
    # encoding_test(tokenized_train_dataset, 'train')
    # encoding_test(tokenized_val_dataset, 'validation')
  
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=trained_model)
    trainer = Seq2SeqTrainer(model=trained_model,
                            args=training_args,
                            train_dataset=tokenized_train_dataset['train'],
                            eval_dataset=tokenized_val_dataset['validation'],
                            data_collator=data_collator,
                            tokenizer=tokenizer,
                            )
    set_seed(training_args.seed)
    trainer.train()
    trainer.save_model('./saved_model_test')
    results = trainer.evaluate()
    print(results)
    trained_model =  AutoModelForSeq2SeqLM.from_pretrained('./saved_model')
    tokenizer = AutoTokenizer.from_pretrained('./saved_model')

    
    # # GENERATION  
    # generated_on_pretrained.py
    
    
    

    

    

    
