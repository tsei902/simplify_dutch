
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, set_seed
# , T5ForConditionalGeneration, TrainingArguments

model_checkpoint = "yhavinga/t5-base-dutch" #"yhavinga/t5-v1.1-base-dutch-cased" #"flax-community/t5-base-dutch"#
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint,  use_cache=False) # gradient_checkpointing=True,
# # REPAIR: model = model.get_device()
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)# , extra_ids=0, additional_special_tokens=0)
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