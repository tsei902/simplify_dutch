# # read both files in

# # get general statistics on both files

# # Which statistics do I need? 
# # get bleu and sentence bleu first? 

#  # human reference translation   - REFERENCE
#  # machine translation          - TRANSLATION
 
#  # how is bleu and sentence bleu measured in code? 
 
# # Sacrebleu - BLEU; chrF, TER
# # human translators score low on bleu


# # METEOR AND COMET BETTER THAN BLEU: 
# # METEOR - (VANdeghinste used NLTK and evaluate but says they are not up to date)
# # https://aclanthology.org/W11-2107.pdf
# # https://www.cs.cmu.edu/~alavie/METEOR/
# # implement this package : https://github.com/wbwseeker/meteor


# # COMET score - pip install unbabel-comet
# # (source, hypothesis, reference)
#    # Machine translation correct if they correlate with human judgement

# #  WER - word error rate 
# # https://pypi.org/project/jiwer/
# # https://jitsi.github.io/jiwer/usage/

# from jiwer import wer, mer, wil
# from comet import download_model, load_from_checkpoint
# import sacrebleu
# import itertools

# from sacrebleu.metrics import BLEU, CHRF, TER
# from sacrebleu.significance import PairedTest, Result, estimate_ci
# from sacrebleu.metrics.base import Metric, Score, Signature
# from utils import read_file, read_lines
# import easse.utils.preprocessing as utils_prep
# import easse.bleu

# import nltk
# from nltk.translate import meteor
# from nltk import word_tokenize, sent_tokenize
# nltk.download('wordnet', quiet=True)
# nltk.download('omw-1.4', quiet=True)
# from utils import yield_lines
# from statistics import mean
# import scipy.stats as stats

# from statsmodels.stats.weightstats import ttest_ind
# import numpy as np
# import pingouin as pg
# from textacy import extract

# if __name__ == '__main__':
#    mt_hypothesis = read_lines('./translations/sample_google_translate.txt')
#    human_ref = read_lines('./translations/sample_human_reference.txt')
#    source = read_lines('./translations/sample_wikilarge.txt')

#    # file_content = open('./translations/sample_wikilarge.txt').read()
#    # sent_sys= nltk.sent_tokenize(file_content, language = "english")
#    # words_sys = nltk.word_tokenize(file_content, language = "english", preserve_line=True)
#    # # print(sent_sys)
#    # # print(words_sys)
#    # print(len(sent_sys))
#    # print(len(words_sys))
#    # # len(human_ref)
#    # # len(sys_sents)
   
#    # file_content_mt = open('./translations/sample_google_translate.txt').read()
#    # sent_sys_mt= nltk.sent_tokenize(file_content_mt, language = "dutch")
#    # words_sys_mt = nltk.word_tokenize(file_content_mt, language = "dutch", preserve_line=True)
#    # # print(sent_sys)
#    # # print(words_sys)
#    # print(len(sent_sys_mt))
#    # print(len(words_sys_mt))
#    # # len(human_ref)
#    # # len(sys_sents)
   
#    # file_content_ht = open('./translations/sample_human_reference.txt').read()
#    # sent_sys_ht= nltk.sent_tokenize(file_content_ht, language = "dutch")
#    # words_sys_ht = nltk.word_tokenize(file_content_ht, language = "dutch", preserve_line=True)
#    # # print(sent_sys)
#    # # print(words_sys)
#    # print(len(sent_sys_ht))
#    # print(len(words_sys_ht))
#    # # len(human_ref)
#    # # len(sys_sents)
#    # print(2247/103)
   
   
#    file_content_ht = open('./resources/datasets/wikilarge/wikilarge.valid.simp', encoding='utf8').read()
#    sent_sys_ht= nltk.sent_tokenize(file_content_ht, language = "dutch")
#    print(sent_sys_ht)
#    words_sys_ht = nltk.word_tokenize(file_content_ht, language = "dutch", preserve_line=True)
#    print(words_sys_ht)
#    lensent = len(sent_sys_ht)
#    lenword= len(words_sys_ht)
#    print(len(sent_sys_ht))
#    print(len(words_sys_ht))
#    # len(human_ref)
#    # len(sys_sents)
#    print(lenword/lensent)