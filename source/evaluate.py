from easse.sari import corpus_sari, get_corpus_sari_operation_scores
from easse.cli import evaluate_system_output
# from source.model import T5FineTuner
from easse.report import get_all_scores
import pandas as pd
import csv
import paths
import utils 
from paths import EXP_DIR, OUTPUT_DIR
# import model
from utils import log_stdout, generate_hash, count_line, read_lines, get_data_filepath
# from preprocess import get_data_filepath
# import time

def evaluate_on_asset(features_kwargs, phase, model_dirname=None):
    dataset = "asset"
    # output_dir = REPO_DIR / f"outputs/{_model_dirname}"
    
    # model_dir =  EXP_DIR / model_dirname #get_last_experiment_dir() if model_dirname is None else
    output_dir = OUTPUT_DIR / "evaluate_on_asset"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Output dir: ", output_dir)
    features_hash = generate_hash(features_kwargs)
    output_score_filepath = output_dir/ f"score_{features_hash}_{dataset}_{phase}_log.txt"
    
    if not output_score_filepath.exists() or count_line(output_score_filepath) == 0:
        # start_time = time.time()
        # complex_filepath = get_data_filepath(dataset, phase, 'orig')
        
        pred_filepath = f'{OUTPUT_DIR}/generate/simplification.txt' # output_dir / f'{features_hash}_{complex_filepath.stem}.txt'
        print('pred filepath', pred_filepath)
        ref_filepaths = [get_data_filepath(dataset, phase,  'simp', i) for i in range(4)]
        
        
        complex_filepath = get_data_filepath(dataset, phase, 'orig')
        print('this is the complex filepath', complex_filepath)
        # if pred_filepath.exists() and count_line(pred_filepath) == count_line(complex_filepath):
        #     print("File is already processed.")
        # else:
        #     model.generate(data, pretrained_model)
            
        # with log_stdout(output_score_filepath):
        # scores = evaluate_system_output(complex_filepath, pred_filepath, [ref_filepaths])
        scores = evaluate_system_output(test_set="asset_test", sys_sents_path=str(pred_filepath), lowercase=True)
        if "WordRatioFeature" in features_kwargs:
            print("W:", features_kwargs["WordRatioFeature"]["target_ratio"], "\t", end="")
        if "CharRatioFeature" in features_kwargs:
            print("C:", features_kwargs["CharRatioFeature"]["target_ratio"], "\t", end="")
        if "LevenshteinRatioFeature" in features_kwargs:
            print("L:", features_kwargs["LevenshteinRatioFeature"]["target_ratio"], "\t", end="")
        if "WordRankRatioFeature" in features_kwargs:
            print("WR:", features_kwargs["WordRankRatioFeature"]["target_ratio"], "\t", end="")
        if "DependencyTreeDepthRatioFeature" in features_kwargs:
            print("DTD:", features_kwargs["DependencyTreeDepthRatioFeature"]["target_ratio"], "\t", end="")
        print("SARI: {:.2f} \t BLEU: {:.2f} \t FKGL: {:.2f} ".format(scores['sari'], scores['bleu'], scores['fkgl']))
        return scores
            # print("Execution time: --- %s seconds ---" % (time.time() - start_time))
    else:
        print("Already exist: ", output_score_filepath)
        print("".join(read_lines(output_score_filepath)))

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
    for i in range(1,len(predictions)):  # range starts at 1 now, source list does not get overwritten
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


