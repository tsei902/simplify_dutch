from jiwer import wer, mer, wil
import sacrebleu
import itertools
from sacrebleu.metrics import BLEU, CHRF, TER
from utils import read_lines
import easse.utils.preprocessing as utils_prep
import nltk
from nltk.translate import meteor
from nltk import word_tokenize
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
from statistics import mean
import scipy.stats as stats
from statsmodels.stats.weightstats import ttest_ind
import numpy as np

if __name__ == '__main__':
   mt_hypothesis = read_lines('./translations/sample_wiki_google_translate.txt')
   human_ref = read_lines('./translations/sample_wiki_human_reference.txt')
   source = read_lines('./translations/sample_wikilarge.txt')

   hyp = mt_hypothesis
   ref = human_ref
   sys_sents = [utils_prep.normalize(sent) for sent in hyp]
   print(type(sys_sents))
   refs_sents = [[utils_prep.normalize(sent) for sent in ref]]  # for ref in ref]
   
   #BLEU
   bleu_test = sacrebleu.corpus_bleu(sys_sents, refs_sents)
   print('bleu from sacrebleu', bleu_test.score)
   
   sys_sents = [utils_prep.normalize(sent) for sent in mt_hypothesis]
   refs_sents = [[utils_prep.normalize(sent) for sent in human_ref]]
   
   bleu_scorer = BLEU() 
   print('bleu corpus score', bleu_scorer.corpus_score(sys_sents, refs_sents,).score)

   # CHRF
   chrf = CHRF()
   chrf_corpus = chrf.corpus_score(sys_sents, refs_sents)
   print(chrf_corpus)

   chrfplus = CHRF(char_order=6,
               word_order=2,
               beta=2,
               lowercase=False,
               whitespace=False,
               eps_smoothing=True)
   print(chrfplus.corpus_score(sys_sents, refs_sents))

   # TER
   ter = TER()
   score = ter.corpus_score(sys_sents, refs_sents)
   print(score)

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
      
   # METEOR: 
   reference = read_lines('./translations/sample_wiki_human_reference.txt')
   candidate = read_lines('./translations/sample_wiki_google_translate.txt')
   meteor_score = []
   for line in zip(reference, candidate):
      reference = line[0]
      candidate = line[1]
      meteor_score.append(round(meteor([word_tokenize(candidate)], word_tokenize(reference)), 4))
   avg_meteor =  mean(meteor_score)
   print('average meteor', avg_meteor)
   
   # SENTENCE LEVEL SCORES 
   refs_sents = [utils_prep.normalize(sent) for sent in ref]
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
      sentence_bleu_list.append(bleu_sent.score)
      chrf_sentence = CHRF(lowercase=True).corpus_score(sys, [ref])
      sentence_chrf_list.append(chrf_sentence.score)
      chrfplus_sentence = CHRF(char_order=6,
               word_order=2,
               beta=2,
               lowercase=False,
               whitespace=False,
               eps_smoothing=True).corpus_score(sys, [ref])
      sentence_chrfplus_list.append(chrfplus_sentence.score)
      sentence_ter = TER(case_sensitive=False).corpus_score(sys, [ref])
      sentence_ter_list.append(sentence_ter.score)
      wer_sentence = wer(sys,ref)
      sentence_wer_list.append(wer_sentence)
      
   # print("sentence bleu", sentence_bleu_list) 
   # print("sentence chrf2", sentence_chrf_list)  
   # print("sentence chrfplus", sentence_chrfplus_list)
   # print("sentence ter", sentence_ter_list)
   # print("sentence wer", sentence_wer_list)
   
   # SENTENCE LEVEL SCORES 
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
      hsentence_bleu_list.append(bleu_sent_human.score)
      chrf_sentence_human = CHRF(lowercase=True).corpus_score(ref, [ref])
      hsentence_chrf_list.append(chrf_sentence_human.score)
      chrfplus_sentence_human = CHRF(char_order=6,
                                    word_order=2,
                                    beta=2,
                                    lowercase=False,
                                    whitespace=False,
                                    eps_smoothing=True).corpus_score(ref, [ref])
      hsentence_chrfplus_list.append(chrfplus_sentence_human.score)
      sentence_ter_human = TER(case_sensitive=False).corpus_score(ref, [ref])
      hsentence_ter_list.append(sentence_ter_human.score)
      wer_sentence_human = wer(ref,ref)
      hsentence_wer_list.append(wer_sentence_human)

  # When the data is perfectly described by the resticted model, the probability to get data that is less well described is 1. 
  # For instance, if the sample means in two groups are identical, the p-values of a t-test is 1. 
   print(stats.ttest_ind(a=hsentence_bleu_list, b=sentence_bleu_list, equal_var=False))
   print(stats.ttest_ind(a=hsentence_chrf_list, b= sentence_chrf_list, equal_var=False))
   # print(stats.ttest_ind(a= sentence_chrf_list,b=hsentence_chrf_list, equal_var=True))
   print(stats.ttest_ind(a= hsentence_chrfplus_list, b= sentence_chrfplus_list, equal_var=False))
   print(stats.ttest_ind(a=hsentence_ter_list, b=sentence_ter_list, equal_var=False))
   print(stats.ttest_ind(a=hsentence_wer_list, b=sentence_wer_list, equal_var=False))
   
   
   # Test for normality - (Formal Statistical Test) Perform a Shapiro-Wilk Test.
   # If the p-value of the test is greater than α = .05, then the data is assumed to be normally distributed.
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
   
   # Result: normality assumption does hold in any of the cases
   
   # create histogram to visualize values in dataset
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
   
   # Result: normality assumption does hold in any of the cases  
   
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
   
   #Wilcoxon Signed Rank Test between refs and hyps
   print(stats.wilcoxon(hsentence_bleu_list,sentence_bleu_list))
   print(stats.wilcoxon(sentence_bleu_list,hsentence_bleu_list))
   print(stats.wilcoxon(hsentence_chrf_list, sentence_chrf_list))
   print(stats.wilcoxon(hsentence_chrfplus_list,sentence_chrfplus_list))
   print(stats.wilcoxon(hsentence_ter_list, sentence_ter_list))
   print(stats.wilcoxon(hsentence_wer_list, sentence_wer_list))
   
   # Output Interpretation:
   
   # https://www.statology.org/wilcoxon-signed-rank-test-python/
   # H0: The mpg is equal between the two groups
   # HA: The mpg is not equal between the two groups

   # Since the p-value (0.001) is less than 0.05, we reject the null hypothesis. 
   # We have sufficient evidence to say that the true mean mpg is not equal between 
   # the two groups.
   
   # https://pythonfordatascienceorg.wordpress.com/wilcoxon-sign-ranked-test-python/
   # The hypothesis being tested is:
   # Null hypothesis (H0): The difference between the pairs follows a symmetric distribution around zero.
   # Alternative hypothesis (HA): The difference between the pairs does not follow a symmetric distribution around zero.
   # If the p-value is less than what is tested at, most commonly 0.05, one can reject the null hypothesis.

   