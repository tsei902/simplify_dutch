from easse.sari import corpus_sari, get_corpus_sari_operation_scores
import pandas as pd
import csv


def calculate_corpus_averages():
    sari_df=  pd.read_csv("./resources/outputs/generate/sari.txt", header=None)
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
    for i in range(1,len(test_dataset['orig'])):  # range starts at 1 now, source list does not get overwritten
        print(i)
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
        print('stat type', type(stat))
        print('stat', stat)
        print('sari', c)
        sari_scores.append(c)
        stats.append(stat)
    with open("resources/outputs/generate/stats.txt", "w", newline='') as f:
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