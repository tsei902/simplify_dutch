# read both files in

# get general statistics on both files

# Which statistics do I need? 
# get bleu and sentence bleu first? 

 # human reference translation   - REFERENCE
 # machine translation          - TRANSLATION
 
 # how is bleu and sentence bleu measured in code? 
 
# Sacrebleu - BLEU; chrF, TER
# human translators score low on bleu


# METEOR AND COMET BETTER THAN BLEU: 
# METEOR - (VANdeghinste used NLTK and evaluate but says they are not up to date)
# https://aclanthology.org/W11-2107.pdf
# https://www.cs.cmu.edu/~alavie/METEOR/
# implement this package : https://github.com/wbwseeker/meteor


# COMET score - pip install unbabel-comet
# (source, hypothesis, reference)
   # Machine translation correct if they correlate with human judgement

#  WER - word error rate 
# https://pypi.org/project/jiwer/
# https://jitsi.github.io/jiwer/usage/

from jiwer import wer, mer, wil
from comet import download_model, load_from_checkpoint
import sacrebleu
import itertools

from sacrebleu.metrics import BLEU, CHRF, TER
from sacrebleu.significance import PairedTest, Result, estimate_ci
from sacrebleu.metrics.base import Metric, Score, Signature
from utils import read_file, read_lines
import easse.utils.preprocessing as utils_prep
import easse.bleu

import nltk
from nltk.translate import meteor
from nltk import word_tokenize
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
from utils import yield_lines
from statistics import mean
import scipy.stats as stats

from statsmodels.stats.weightstats import ttest_ind
import numpy as np
import pingouin as pg
from textacy import extract

if __name__ == '__main__':
   
   mt_hypothesis = read_lines('./translations/sample_wiki_google_translate.txt')
   # print(mt_hypothesis)
   human_ref = read_lines('./translations/sample_wiki_human_reference.txt')
   # print(human_ref)
   source = read_lines('./translations/sample_wikilarge.txt')
   # print(source)

   hyp = mt_hypothesis
   ref = human_ref
   sys_sents = [utils_prep.normalize(sent) for sent in hyp]
   # print('SYS', sys_sents)
   print(type(sys_sents))
   refs_sents = [[utils_prep.normalize(sent) for sent in ref]]  # for ref in ref]
   # print('REF', refs_sents)
   bleu_test = sacrebleu.corpus_bleu(sys_sents, refs_sents)
   # print(bleu_test)
   print('bleu from sacrebleu', bleu_test.score)
 
   # SACREBLEU:
   # @ from easse:
   # the normalize package makes lowercase
   # tokenizer 13a
   # no detokenization used
   
   sys_sents = [utils_prep.normalize(sent) for sent in mt_hypothesis]
   refs_sents = [[utils_prep.normalize(sent) for sent in human_ref]]
   
   bleu_scorer = BLEU() # lowercase=False, force=True,
                     # smooth_method="exp", smooth_value=None, effective_order=False)
   # print('scores', bleu_scorer.corpus_score(sys_sents, refs_sents))
   print('bleu corpus score', bleu_scorer.corpus_score(sys_sents, refs_sents,).score)

   # print(bleu_scorer.get_signature())
   # print(bleu_scorer._aggregate_and_compute)

   # fgkl_scorer = FGKL() # lowercase=False, force=True,
                     # smooth_method="exp", smooth_value=None, effective_order=False)
   # print('scores', bleu_scorer.corpus_score(sys_sents, refs_sents))
   # print('fgkl corpus score', bleu_scorer.corpus_score(sys_sents, refs_sents,).score)
   # bleu needs to be stripped from truecasing and tokenization!
   # https://bricksdont.github.io/posts/2020/12/computing-and-reporting-bleu-scores/

   # bleu needs:
   # lowercase
   # detokenized data? bleu works once on tokenized, once on detokenized data
   # detokenizer not used in cheng sheang

   # lowercase: used in stopword removal
   # lowercase is always set to true - what does that mean?
   # is the input tokenized? thus individual words? no input is handled as a string
   # interpretation of the bleu score:  # the numbers from SacreBLEU are already multiplied by 100, unlike NLTK.
   # DO I need more data?
   
   # 29.44 82.4/42.9/27.3/12.5 (BP = 0.889 ratio = 0.895 hyp_len = 17 ref_len = 19)
   # 29.44 refers to the final BLEU score
   # 82.4/42.9/27.3/12.5 represents the precision value for 1–4 ngram order
   # BP is the brevity penalty
   # ratio indicates the ratio between hypothesis length and reference length
   # hyp_len refers to the total number of characters for hypothesis text
   # ref_len is the total number of characters for reference text
   
   

   # CHRF
   chrf = CHRF()
   chrf_corpus = chrf.corpus_score(sys_sents, refs_sents)
   print(chrf_corpus)
   # maybe set chrf to word order 1 - see Popovic paper

   chrfplus = CHRF(char_order=6,
               word_order=2,
               beta=2,
               lowercase=False,
               whitespace=False,
               eps_smoothing=True)
   print(chrfplus.corpus_score(sys_sents, refs_sents))
   # sentence sentence_chrf, sentence TER, chrf++
   # the higher the better
   # for specific details, read up on some papers!
   # unsure if eps smoothin needs to be true!

   # TER
   #  remove punctuation?
   # apply normalizaiton and tokenization?
   # the lower the better
   ter = TER()
   score = ter.corpus_score(sys_sents, refs_sents)
   print(score)
   # result = Result(score)

   # # SIGNIFICANCE LEVELS for BLEU, chrf, TER
   # manual hacking of significance scores: 
   # #https://colab.research.google.com/drive/15nI8tPIhxBoLi4AjdnfoK_ERLU49xf5V?usp=sharing#scrollTo=zOqBe5DEdrix
   
   # !sacrebleu REFTXTREF.txt.sent -i ORHERFILEHERE.txt  -m bleu ter chrf --paired-bs
      
   # on sentence level!! 
   # WER
   error = wer(human_ref, mt_hypothesis)
   print('word error rate (wer): ', error)
   
   hum_ref_low = list(map(lambda x: x.lower(), human_ref))
   mt_hyp_low =  list(map(lambda y: y.lower(), mt_hypothesis))
   error_lower = wer(hum_ref_low,mt_hyp_low)
   print('word error rate (wer): ', error_lower)

   
   mer = mer(human_ref, mt_hypothesis)
   print('Match error rate (mer): ', mer)
   
   wil = wil(human_ref, mt_hypothesis)
   print('word information lost (wil): ', wil)
   
   # https://www.researchgate.net/publication/221478089_From_WER_and_RIL_to_MER_and_WIL_improved_evaluation_measures_for_connected_speech_recognition
   # scores should be below 1
   # The lower the value, the better the performance of the ASR system, with a WER of 0 being a perfect score.
   
      
   # METEOR: 
   # from nltk - do not compare with publications
   # https://blog.machinetranslation.io/compute-bleu-score/
   reference = read_lines('./translations/sample_wiki_human_reference.txt')
   candidate = read_lines('./translations/sample_wiki_google_translate.txt')
   
   meteor_score = []
   for line in zip(reference, candidate):
      reference = line[0]
      candidate = line[1]
      meteor_score.append(round(meteor([word_tokenize(candidate)], word_tokenize(reference)), 4))# list of references
   
   avg_meteor =  mean(meteor_score)
   print('average meteor', avg_meteor)
   
   # value is between 1 and 0
   # stemmer: StemmerI = PorterStemmer(),
   # wordnet: WordNetCorpusReader = wordnet,
   # alpha: float = 0.9,
   # beta: float = 3.0,
   # gamma: float = 0.5,
   
   
   # SENTENCE LEVEL SCORES 
   
   refs_sents = [utils_prep.normalize(sent) for sent in ref]
   # print(refs_sents[1:5])
   sentence_bleu_list = []
   sentence_bleu_list2 = []
   sentence_chrf_list = []
   sentence_chrfplus_list = []
   sentence_ter_list = []
   sentence_wer_list = []
   sentence_mer_list = []
   sentence_wil_list = []
   
   for (sys,ref) in itertools.zip_longest(sys_sents, refs_sents): 
      bleu_sent = sacrebleu.sentence_bleu(sys,[ref])
      # print('sentece score', bleu_sent.score)
      sentence_bleu_list.append(bleu_sent.score)
      chrf_sentence = CHRF(lowercase=True).corpus_score(sys, [ref])
      sentence_chrf_list.append(chrf_sentence.score)
      # print(chrf_sentence.score)
      chrfplus_sentence = CHRF(char_order=6,
               word_order=2,
               beta=2,
               lowercase=False,
               whitespace=False,
               eps_smoothing=True).corpus_score(sys, [ref])
      sentence_chrfplus_list.append(chrfplus_sentence.score)
      sentence_ter = TER(case_sensitive=False).corpus_score(sys, [ref])
      # print(sentence_ter.score)
      sentence_ter_list.append(sentence_ter.score)
      wer_sentence = wer(sys,ref)
      # print(wer_sentence)
      sentence_wer_list.append(wer_sentence)
      # mer_sentence = mer(sys,ref)
      # print(mer_sentence)
      # sentence_mer_list.append(mer_sentence)
      # wil_sentence = wil(sys,ref)
      # print(wil_sentence)
      # sentence_wil_list.append(wil_sentence)

      
   print("sentence bleu", sentence_bleu_list)
   # print(sentence_bleu_list2) 
   print("sentence chrf2", sentence_chrf_list)  # Chrf2? 
   print("sentence chrfplus", sentence_chrfplus_list)
   print("sentence ter", sentence_ter_list)# Chrf2? 
   print("sentence wer", sentence_wer_list)# Chrf2? 
   # meteor_score is already a sentence-level string
   
   
      # SENTENCE LEVEL SCORES 
   
   # refs_sents = [utils_prep.normalize(sent) for sent in ref]
   # print(refs_sents[1:5])
   hsentence_bleu_list = []
   hsentence_bleu_list2 = []
   
   hsentence_chrf_list = []
   hsentence_chrfplus_list = []
   hsentence_ter_list = []
   hsentence_wer_list = []
   hsentence_mer_list = []
   hsentence_wil_list = []
   
   for (ref,ref) in itertools.zip_longest(refs_sents, refs_sents): 
      bleu_sent_human = sacrebleu.sentence_bleu(ref,[ref])
      # print('sentece score', bleu_sent.score)
      hsentence_bleu_list.append(bleu_sent_human.score)
      chrf_sentence_human = CHRF(lowercase=True).corpus_score(ref, [ref])
      
      hsentence_chrf_list.append(chrf_sentence_human.score)
      # print(chrf_sentence.score)
      chrfplus_sentence_human = CHRF(char_order=6,
                                    word_order=2,
                                    beta=2,
                                    lowercase=False,
                                    whitespace=False,
                                    eps_smoothing=True).corpus_score(ref, [ref])
      hsentence_chrfplus_list.append(chrfplus_sentence_human.score)
      sentence_ter_human = TER(case_sensitive=False).corpus_score(ref, [ref])
      # print(sentence_ter.score)
      hsentence_ter_list.append(sentence_ter_human.score)
      wer_sentence_human = wer(ref,ref)
      # print(wer_sentence)
      hsentence_wer_list.append(wer_sentence_human)
      # mer_sentence = mer(sys,ref)
      # print(mer_sentence)
      # sentence_mer_list.append(mer_sentence)
      # wil_sentence = wil(sys,ref)
      # print(wil_sentence)
      # sentence_wil_list.append(wil_sentence)


    # print("hsentence bleu", hsentence_bleu_list) # all 100
   # print("hsentence chrf2", hsentence_chrf_list)  # Chrf2? # all 100
   # print("hsentence chrfplus", hsentence_chrfplus_list) # all 25
   # print("hsentence ter", hsentence_ter_list) # all 0.0 
   # print("hsentence wer", hsentence_wer_list) # all 0.0 
   # meteor_score is already a sentence-level string

   # t test assumes normality
   
   # print(pg.ttest(hsentence_bleu_list,sentence_bleu_list, correction=True))
   # # homoheneity assumption
   
   # print(pg.ttest(hsentence_bleu_list,sentence_bleu_list, correction=False))
   # # homoheneity assumption
  
   
   # print(ttest_ind(hsentence_bleu_list,sentence_bleu_list))
   # # tstat, pvalue, df
   
  # Yes. When the data is perfectly described by the resticted model, the probability to get data that is less well described is 1. For instance, if the sample means in two groups are identical, the p-values of a t-test is 1
   print(stats.ttest_ind(a=hsentence_bleu_list, b=sentence_bleu_list, equal_var=False))
   print(stats.ttest_ind(a=hsentence_chrf_list, b= sentence_chrf_list, equal_var=False))
   # print(stats.ttest_ind(a= sentence_chrf_list,b=hsentence_chrf_list, equal_var=True))
   print(stats.ttest_ind(a= hsentence_chrfplus_list, b= sentence_chrfplus_list, equal_var=False))
   print(stats.ttest_ind(a=hsentence_ter_list, b=sentence_ter_list, equal_var=False))
   print(stats.ttest_ind(a=hsentence_wer_list, b=sentence_wer_list, equal_var=False))
   
   
   # test for normality - (Formal Statistical Test) Perform a Shapiro-Wilk Test.
   # If the p-value of the test is greater than α = .05, then the data is assumed to be normally distributed.
   import math
   import numpy as np
   from scipy.stats import shapiro 
   import matplotlib.pyplot as plt
   
   #make this example reproducible
   np.random.seed(1)

   #perform Shapiro-Wilk test for normality
   print("shapiro bleu", shapiro(hsentence_bleu_list))
   print("shapiro crf", shapiro(hsentence_chrf_list))
   print("shapiro crf plus", shapiro(hsentence_chrfplus_list))
   print("shapiro ter", shapiro(hsentence_ter_list))
   print("shapiro wer", shapiro(hsentence_wer_list))
   
   # normality assumption does hold in any of the cases
   
   # shapiro bleu ShapiroResult(statistic=1.0, pvalue=1.0)
   # shapiro crf ShapiroResult(statistic=1.0, pvalue=1.0)
   # shapiro crf plus ShapiroResult(statistic=1.0, pvalue=1.0)
   # shapiro ter ShapiroResult(statistic=1.0, pvalue=1.0)
   # shapiro wer ShapiroResult(statistic=1.0, pvalue=1.0)
   
   #create histogram to visualize values in dataset
   # plt.hist(hsentence_bleu_list, edgecolor='black', bins=20)
   # plt.hist(hsentence_chrf_list, edgecolor='black', bins=20)
   # plt.hist(hsentence_chrfplus_list, edgecolor='black', bins=20)
   # plt.hist(hsentence_ter_list, edgecolor='black', bins=20)
   # plt.hist(hsentence_wer_list, edgecolor='black', bins=20)
   
   
   # perform Shapiro-Wilk test for normality
   print("shapiro bleu", shapiro(sentence_bleu_list))
   print("shapiro crf", shapiro(sentence_chrf_list))
   print("shapiro crf plus", shapiro(sentence_chrfplus_list))
   print("shapiro ter", shapiro(sentence_ter_list))
   print("shapiro wer", shapiro(sentence_wer_list))
   
   # shapiro bleu ShapiroResult(statistic=0.9249647259712219, pvalue=2.3866128685767762e-05)
   # shapiro crf ShapiroResult(statistic=0.8613736629486084, pvalue=2.835239421017377e-08)
   # shapiro crf plus ShapiroResult(statistic=0.8613736629486084, pvalue=2.835239421017377e-08)
   # shapiro ter ShapiroResult(statistic=0.857096791267395, pvalue=1.938811777790761e-08)
   # shapiro wer ShapiroResult(statistic=0.907876193523407, pvalue=3.064859129153774e-06)
   
   
   #create histogram to visualize values in dataset
   plt.hist(sentence_bleu_list, edgecolor='black', bins=40)
   #plt.show()
   plt.hist(sentence_chrf_list, edgecolor='black', bins=40)
   #plt.show()
   plt.hist(sentence_chrfplus_list, edgecolor='black', bins=40)
   #plt.show()
   plt.hist(sentence_ter_list, edgecolor='black', bins=40)
   #plt.show()
   plt.hist(sentence_wer_list, edgecolor='black', bins=40)
   #plt.show()
   
   #Wilcoxon Signed Rank Test
   
   print(stats.wilcoxon(hsentence_bleu_list,sentence_bleu_list))
   print(stats.wilcoxon(sentence_bleu_list,hsentence_bleu_list))
   print(stats.wilcoxon(hsentence_chrf_list, sentence_chrf_list))
   print(stats.wilcoxon(hsentence_chrfplus_list,sentence_chrfplus_list))
   print(stats.wilcoxon(hsentence_ter_list, sentence_ter_list))
   print(stats.wilcoxon(hsentence_wer_list, sentence_wer_list))
   
   # WilcoxonResult(statistic=0.0, pvalue=5.2793130774189556e-14)
   # WilcoxonResult(statistic=0.0, pvalue=7.732733053009578e-14)
   # WilcoxonResult(statistic=0.0, pvalue=7.732733053009578e-14)
   # WilcoxonResult(statistic=0.0, pvalue=7.732733053009578e-14)
   # WilcoxonResult(statistic=0.0, pvalue=5.268196710505936e-14)
   
   # WilcoxonResult(statistic=2234.5, pvalue=0.0014107333565442858)
   # Output Interpretation:

   # In the above example, the p-value is 0.001 which is less than the threshold(0.05) 
   # which is the alpha(0.05) i.e. p-value<alpha which means the sample 
   # is of the same distribution and the sample distributions are equal 
   # if in the case if the p-value>0.05 than it would be opposite.
   
   
   # https://www.statology.org/wilcoxon-signed-rank-test-python/
   # H0: The mpg is equal between the two groups

   # HA: The mpg is not equal between the two groups

   # Since the p-value (0.044) is less than 0.05, we reject the null hypothesis. 
   # We have sufficient evidence to say that the true mean mpg is not equal between 
   # the two groups.
   
   # https://pythonfordatascienceorg.wordpress.com/wilcoxon-sign-ranked-test-python/
   # The hypothesis being test is:

   # Null hypothesis (H0): The difference between the pairs follows a symmetric distribution around zero.
   # Alternative hypothesis (HA): The difference between the pairs does not follow a symmetric distribution around zero.
   # If the p-value is less than what is tested at, most commonly 0.05, one can reject the null hypothesis.
   
   # no equal distribution of means in any case.
   