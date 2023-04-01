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

def evaluate_on_dataset(features_kwargs, model_dirname, eval_dataset): #, phase): # model_dirname=None):
    wandb.init(project= "Tokens_tuning", job_type="evaluation_SARI")
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
        if "WordRatioFeature" in features_kwargs:
            print("W:", "%.2f" % features_kwargs["WordRatioFeature"]["target_ratio"], "\t", end="")
        if "CharRatioFeature" in features_kwargs:
            print("C:", "%.2f" % features_kwargs["CharRatioFeature"]["target_ratio"], "\t", end="")
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
        if "WordRatioFeature" in features_kwargs:
            print("W:", "%.2f" % features_kwargs["WordRatioFeature"]["target_ratio"], "\t", end="")
        if "CharRatioFeature" in features_kwargs:
            print("C:", "%.2f" % features_kwargs["CharRatioFeature"]["target_ratio"], "\t", end="")
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
        if "WordRatioFeature" in features_kwargs:
            print("W:", "%.2f" % features_kwargs["WordRatioFeature"]["target_ratio"], "\t", end="")
        if "CharRatioFeature" in features_kwargs:
            print("C:", "%.2f" % features_kwargs["CharRatioFeature"]["target_ratio"], "\t", end="")
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

    #######NOT NEEDED ATM!!
    # PREPROCESS TEST DATA (ASSET and WIKILARGE) 
    # preprocessor = Preprocessor(features)
    # preprocessor.preprocess_dataset(ASSET_DATASET)
    # # 2) prepare and tokenize 
    # test_dataset = prepare.get_test_data(ASSET_PROCESSED, 0, 358) # doesnt take first row.
    # print('test_dataset', test_dataset)