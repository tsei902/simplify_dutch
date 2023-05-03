# # Corpus BLEU with arguments
# # Run this file from CMD/Terminal
# # Example Command: python3 compute-bleu-args.py test_file_name.txt mt_file_name.txt


# import sys
# import sacrebleu
# from sacremoses import MosesDetokenizer
# md = MosesDetokenizer(lang='nl')

# #   mt_hypothesis = read_lines('./translations/sample_wiki_google_translate.txt')
# #    # print(mt_hypothesis)
# #    human_ref = read_lines('./translations/sample_wiki_human_reference.txt')

# # Open the test dataset human translation file and detokenize the references
# refs = []

# with open('./translations/sample_asset_human_reference.txt') as test:
#     for line in test: 
#         line = line.strip().split() 
#         line = md.detokenize(line) 
#         refs.append(line)
    
# print("Reference 1st sentence:", refs[0])

# refs = [refs]  # Yes, it is a list of list(s) as required by sacreBLEU


# # Open the translation file by the NMT model and detokenize the predictions
# preds = []

# with open('./translations/sample_asset_google_translate.txt') as pred:  
#     for line in pred: 
#         line = line.strip().split() 
#         line = md.detokenize(line) 
#         preds.append(line)

# print("MTed 1st sentence:", preds[0])    


# # Calculate and print the BLEU score
# bleu = sacrebleu.corpus_bleu(preds, refs, lowercase=True)
# print("BLEU: ", bleu.score)

# # from: https://blog.machinetranslation.io/compute-bleu-score/#detoc