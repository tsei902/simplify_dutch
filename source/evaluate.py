import easse
from easse.sari import corpus_sari, get_corpus_sari_operation_scores
from easse.cli import evaluate_system_output
from easse.utils.constants import ( VALID_METRICS, DEFAULT_METRICS)
import pandas as pd
import csv
import wandb
import paths
import utils 
from utils import generate_hash, count_line, read_lines_ref
from model import simplify
from paths import REPO_DIR, ASSET_DATASET, get_data_filepath,  OUTPUT_DIR, WIKILARGE_DATASET
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from paths import ASSET_DATASET

DEFAULT_METRICS = ['bleu', 'sari', 'fkgl', 'sent_bleu', 'f1_token', 'sari_by_operation']

def evaluate_on_dataset(features_kwargs, model_dirname, eval_dataset, project_name): #, phase): # model_dirname=None):
    wandb.init(project= project_name, job_type="evaluation_SARI")
    if eval_dataset== ASSET_DATASET: 
        dataset= "asset_test"
    model_dir = REPO_DIR /f"{model_dirname}"
    print('model dir', model_dir)
    pretrained_model =  AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    output_dir = OUTPUT_DIR / "evaluate_on_dataset"
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Output dir: ", output_dir)
    pred_filepath = f'{OUTPUT_DIR}/evaluate_on_dataset/simp_{wandb.run.name}.txt'
    orig_pfad = get_data_filepath(eval_dataset, 'test', 'orig') 
    # ref_pfad = get_data_filepath(dataset, phase, 'simple')
    ref_filepaths = [get_data_filepath(eval_dataset, 'test', 'simp', i) for i in range(10)]
    simplify(orig_pfad, pretrained_model, tokenizer, features_kwargs, output_folder=pred_filepath )
    for i in range(len(pred_filepath)):
        scores = evaluate_system_output(test_set="custom", orig_sents_path=orig_pfad, sys_sents_path=str(pred_filepath), refs_sents_paths= ref_filepaths,  lowercase=True,metrics = DEFAULT_METRICS)
        if "CharLengthRatioFeature" in features_kwargs:
            print("C:", "%.2f" % features_kwargs["CharLengthRatioFeature"]["target_ratio"], "\t", end="")
        if "WordLengthRatioFeature" in features_kwargs:
            print("W:", "%.2f" % features_kwargs["WordLengthRatioFeature"]["target_ratio"], "\t", end="")
        if "LevenshteinRatioFeature" in features_kwargs:
            print("L:", "%.2f" % features_kwargs["LevenshteinRatioFeature"]["target_ratio"], "\t", end="")
        if "WordRankRatioFeature" in features_kwargs:
            print("WR:","%.2f" % features_kwargs["WordRankRatioFeature"]["target_ratio"], "\t", end="")
        if "DependencyTreeDepthRatioFeature" in features_kwargs:
            print("DTD:", "%.2f" % features_kwargs["DependencyTreeDepthRatioFeature"]["target_ratio"], "\t", end="")
        print("SARI: {:.2f} \t BLEU: {:.2f} \t FKGL: {:.2f} \t SENT_BLEU: {:.2f} \t F1 {:.2f}  \t SARI_ADD {:.2f} \t SARI_KEEP {:.2f} \t SARI_DELETE {:.2f}".format(scores['sari'], scores['bleu'], scores['fkgl'], scores['sent_bleu'],  scores['f1_token'],  scores['sari_add'], scores['sari_keep'], scores['sari_del'])) # test
        # write lines into output dir 
        print(scores)
        wandb.log(scores)
        return  scores['sari']
    
def evaluate_on_asset(features_kwargs, model_dirname, eval_dataset): #, phase): # model_dirname=None):
    # wandb.init(project= "Tokens_tuning", job_type="evaluation_SARI")
    if eval_dataset== ASSET_DATASET: 
        dataset= "asset_test"
    model_dir = REPO_DIR /f"{model_dirname}"
    print(model_dir)
    pretrained_model =  AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    # output_dir = OUTPUT_DIR / "evaluate_on_asset"
    # output_dir.mkdir(parents=True, exist_ok=True)
    # print("Output dir: ", output_dir)
    # features_hash = generate_hash(features_kwargs)

    pred_filepath = f'{OUTPUT_DIR}/generate/simplification.txt' 
    orig_pfad = get_data_filepath(eval_dataset, 'test', 'orig') 
    
    if pred_filepath and count_line(pred_filepath) == count_line(orig_pfad):
        print("File is already processed.")
    else:
        simplify(orig_pfad, pretrained_model, tokenizer, features_kwargs)
    for i in range(len(pred_filepath)):
        scores = evaluate_system_output(test_set="asset_test", sys_sents_path=str(pred_filepath), lowercase=True,metrics = DEFAULT_METRICS)
        if "CharLengthRatioFeature" in features_kwargs:
            print("C:", "%.2f" % features_kwargs["CharLengthRatioFeature"]["target_ratio"], "\t", end="")
        if "WordLengthRatioFeature" in features_kwargs:
            print("W:", "%.2f" % features_kwargs["WordLengthRatioFeature"]["target_ratio"], "\t", end="")
        if "LevenshteinRatioFeature" in features_kwargs:
            print("L:", "%.2f" % features_kwargs["LevenshteinRatioFeature"]["target_ratio"], "\t", end="")
        if "WordRankRatioFeature" in features_kwargs:
            print("WR:","%.2f" % features_kwargs["WordRankRatioFeature"]["target_ratio"], "\t", end="")
        if "DependencyTreeDepthRatioFeature" in features_kwargs:
            print("DTD:", "%.2f" % features_kwargs["DependencyTreeDepthRatioFeature"]["target_ratio"], "\t", end="")
        print("SARI: {:.2f} \t BLEU: {:.2f} \t FKGL: {:.2f} \t SENT_BLEU: {:.2f} \t F1 {:.2f}  \t SARI_ADD {:.2f} \t SARI_KEEP {:.2f} \t SARI_DELETE {:.2f}".format(scores['sari'], scores['bleu'], scores['fkgl'], scores['sent_bleu'],  scores['f1_token'],  scores['sari_add'], scores['sari_keep'], scores['sari_del'])) # test
        # write lines into output dir 
        # print(scores)
        # wandb.log(scores)
        return  scores['sari']

def evaluate_corpus(features_kwargs): 
    pred_filepath = f'{OUTPUT_DIR}/generate/simplification.txt'
    print('pred filepath', pred_filepath)
    for i in range(len(pred_filepath)): 
        scores = evaluate_system_output(test_set="asset_test", sys_sents_path=str(pred_filepath), lowercase=True, metrics = DEFAULT_METRICS) # VALID_METRICS)
        if "CharLengthRatioFeature" in features_kwargs:
            print("C:", "%.2f" % features_kwargs["CharLengthRatioFeature"]["target_ratio"], "\t", end="")
        if "WordLengthRatioFeature" in features_kwargs:
            print("W:", "%.2f" % features_kwargs["WordLengthRatioFeature"]["target_ratio"], "\t", end="")
        if "LevenshteinRatioFeature" in features_kwargs:
            print("L:", "%.2f" % features_kwargs["LevenshteinRatioFeature"]["target_ratio"], "\t", end="")
        if "WordRankRatioFeature" in features_kwargs:
            print("WR:","%.2f" % features_kwargs["WordRankRatioFeature"]["target_ratio"], "\t", end="")
        if "DependencyTreeDepthRatioFeature" in features_kwargs:
            print("DTD:", "%.2f" % features_kwargs["DependencyTreeDepthRatioFeature"]["target_ratio"], "\t", end="")
        print("SARI: {:.2f} \t BLEU: {:.2f} \t FKGL: {:.2f} \t SENT_BLEU: {:.2f} \t F1 {:.2f}  \t SARI_ADD {:.2f} \t SARI_KEEP {:.2f} \t SARI_DELETE {:.2f}".format(scores['sari'], scores['bleu'], scores['fkgl'], scores['sent_bleu'],  scores['f1_token'],  scores['sari_add'], scores['sari_keep'], scores['sari_del'])) # test
        # wandb.log({"Scores": scores})
        # print("Execution time: --- %s seconds ---" % (time.time() - start_time))
        return scores['sari']
    
def calculate_corpus_averages():
    sari_df=  pd.read_csv(f'{OUTPUT_DIR}/generate/sari.txt', header=None)
    avg_sari = sari_df.mean().item()
    print('sari_average', avg_sari)
    df = pd.read_csv("./resources/outputs/generate/stats.txt")    
    avg_add= df['add'].mean().item()
    avg_keep = df['keep'].mean().item()
    avg_delete = df['del'].mean().item()
    # print('avgadd', avg_add)
    # print('averages: ', avg_add, avg_keep, avg_delete)
    return avg_sari , avg_add, avg_keep, avg_delete

def calculate_eval_sentence(dataset, test_dataset, predictions):
    sari_scores = []
    stats = []
    print('len predictions' , len(predictions)) # 
    print('len orig sentences', len(test_dataset['orig']))
    for i in range(0,len(predictions)):  # range starts at 1 now, source list does not get overwritten
        # print(i)
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
        # print('stat type', type(stat))
        # print('stat', stat)
        # print('sari', c)
        sari_scores.append(c)
        stats.append(stat)
    with open(f'{OUTPUT_DIR}/generate/stats.txt', "w", newline='') as f:
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
    
    
if __name__ == '__main__':
    orig_pfad = get_data_filepath(ASSET_DATASET, 'test', 'orig') 
    # ref_pfad = get_data_filepath(dataset, phase, 'simple')
    ref_filepaths = [get_data_filepath(ASSET_DATASET, 'test', 'simp', i) for i in range(10)]
    pred_filepath = f'{OUTPUT_DIR}/final_decoder_outputs/beampk120099repearly_full.txt' # f'{OUTPUT_DIR}/evaluate_on_dataset/simplification.txt'   #  './resources/datasets/asset/asset.test.simp.3' # f'{OUTPUT_DIR}/evaluate_on_dataset/asset.test.simp.4'
    
    scores = evaluate_system_output(test_set="custom", orig_sents_path=orig_pfad, sys_sents_path=str(pred_filepath), refs_sents_paths= ref_filepaths,  lowercase=True,metrics = DEFAULT_METRICS)
    print("SARI: {:.2f} \t BLEU: {:.2f} \t FKGL: {:.2f} ".format(scores['sari'], scores['bleu'], scores['fkgl'])) # test
    print(scores)
    
    
    
# greedy full
# SARI: 36.26      BLEU: 83.37     FKGL: 8.30 
# {'bleu': 83.36991192907412, 'sent_bleu': 76.91567444867039, 'sari': 36.25526154754177, 'sari_add': 2.049787249600881, 'sari_keep': 54.936704728635085, 'sari_del': 51.779292664389345, 'fkgl': 8.304279903497633, 'f1_token': 79.43104343824497}


# topp topk 5 0.98: 
# SARI: 38.04      BLEU: 66.16     FKGL: 7.84 
# {'bleu': 66.16062722085984, 'sent_bleu': 58.00580130968169, 'sari': 38.03973399700609, 'sari_add': 3.260228525312191, 'sari_keep': 49.561308255719375, 'sari_del': 61.2976652099867, 'fkgl': 7.8367076773578574, 'f1_token': 70.67985114259491}
    
# beampk: 
# SARI: 33.73      BLEU: 87.14     FKGL: 9.04 
# {'bleu': 87.1403809517648, 'sent_bleu': 83.21604292774069, 'sari': 33.72609466874248, 'sari_add': 1.673646005633974, 'sari_keep': 56.71907118230784, 'sari_del': 42.78556681828563, 'fkgl': 9.04440306542308, 'f1_token': 84.04786499067309}
    
    
    
    
# simp beampk. 
# SARI: 36.85      BLEU: 83.38     FKGL: 8.05 
# {'bleu': 83.38051696425413, 'sent_bleu': 77.84182881473753, 'sari': 36.851028817290334, 'sari_add': 2.2677019741616298, 'sari_keep': 54.140992153963516, 'sari_del': 54.14439232374586, 'fkgl': 8.05050861634152, 'f1_token': 79.68003556227}

# simp greedy
# SARI: 19.79      BLEU: 90.11     FKGL: 10.75 
# {'bleu': 90.11244498755592, 'sent_bleu': 89.058650453152, 'sari': 19.794138894201456, 'sari_add': 0.0, 'sari_keep': 59.382416682604365, 'sari_del': 0.0, 'fkgl': 10.751382398340006, 'f1_token': 91.29973470116627}

# greedy 50 len
# SARI: 36.12      BLEU: 83.35     FKGL: 8.36 
# {'bleu': 83.35401982985516, 'sent_bleu': 76.99754245865779, 'sari': 36.119819113207484, 'sari_add': 2.05232971563062, 'sari_keep': 55.220297967737196, 'sari_del': 51.086829656254636, 'fkgl': 8.36109788781857, 'f1_token': 79.75758737436077}

# SARI: 36.26      BLEU: 83.37     FKGL: 8.30 
# {'bleu': 83.36991192907412, 'sent_bleu': 76.91567444867039, 'sari': 36.25526154754177, 'sari_add': 2.049787249600881, 'sari_keep': 54.936704728635085, 'sari_del': 51.779292664389345, 'fkgl': 8.304279903497633, 'f1_token': 79.43104343824497}

# top3pk98: 
# SARI: 37.88      BLEU: 66.26     FKGL: 7.53 
# {'bleu': 66.26095306789612, 'sent_bleu': 58.93274229474616, 'sari': 37.88302862965144, 'sari_add': 3.1950348992683764, 'sari_keep': 49.130450707673994, 'sari_del': 61.32360028201194, 'fkgl': 7.529030236574126, 'f1_token': 71.8028383853241}

# top5pk98:
# SARI: 37.85      BLEU: 65.73     FKGL: 7.74 
# {'bleu': 65.72765337098929, 'sent_bleu': 59.01359520319995, 'sari': 37.85094906035159, 'sari_add': 3.2738941293037893, 'sari_keep': 49.40083759497595, 'sari_del': 60.87811545677504, 'fkgl': 7.738092602272808, 'f1_token': 70.63999632444539}



# Asset test 4 
# SARI: 53.20      BLEU: 100.00    FKGL: 7.28 'bleu': 100.00000000000004, 'sent_bleu': 100.00000000000001, 'sari': 53.20372185554857, 'sari_add': 24.716643929006167, 'sari_keep': 62.94411666953717, 'sari_del': 71.95040496810239, 'fkgl': 7.276940392951033, 'f1_token': 100.0  

# Asset test 2
# SARI: 52.86      BLEU: 100.00    FKGL: 7.34 'bleu': 100.00000000000004, 'sent_bleu': 100.00000000000001, 'sari': 52.85685196669839, 'sari_add': 25.529519203466734, 'sari_keep': 60.06554676955884, 'sari_del': 72.97548992706959, 'fkgl': 7.3386123092743425, 'f1_token': 100.0

# Orig file: SARI: 19.79      BLEU: 90.11     FKGL: 10.75 'bleu': 90.11244498755592, 'sent_bleu': 89.058650453152, 'sari': 19.794138894201456, 'sari_add': 0.0, 'sari_keep': 59.382416682604365, 'sari_del': 0.0, 'fkgl': 10.751382398340006, 'f1_token': 91.29973470116627

