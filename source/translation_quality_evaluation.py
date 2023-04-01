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

from sacrebleu.metrics import BLEU, CHRF, TER
from sacrebleu.significance import PairedTest, Result, estimate_ci
from sacrebleu.metrics.base import Metric, Score, Signature
from utils import read_file, read_lines
import easse.utils.preprocessing as utils_prep
import easse.bleu

import nltk
from nltk.translate import meteor
from nltk import word_tokenize
nltk.download('wordnet')
nltk.download('omw-1.4')
from utils import yield_lines
from statistics import mean

if __name__ == '__main__':
   mt_hypothesis = read_lines('./translations/sample_google_translate.txt')
   # print(mt_hypothesis)
   human_ref = read_lines('./translations/sample_human_reference.txt')
   # print(human_ref)
   source = read_lines('./translations/sample_wikilarge.txt')
   # print(source)

   hyp = human_ref
   ref = human_ref
   sys_sents = [utils_prep.normalize(sent) for sent in hyp]
   # print('SYS', sys_sents)
   refs_sents = [[utils_prep.normalize(sent)
                                       for sent in ref]]  # for ref in ref]
   # print('REF', refs_sents)
   bleu_test = sacrebleu.corpus_bleu(sys_sents, refs_sents)
   print(bleu_test)
   
   print('test', bleu_test.score)
   # print(bleu_test.estimate_ci(int(bleu_test.score)))
   
   # SACREBLEU:
   # @ from easse:
   # the normalize package makes lowercase
   # tokenizer 13a
   # no detokenization used
   sys_sents = [utils_prep.normalize(sent) for sent in mt_hypothesis]
   # for human_ref in human_ref]
   refs_sents = [[utils_prep.normalize(sent) for sent in human_ref]]

   bleu_scorer = BLEU(lowercase=False, force=True,
                     smooth_method="exp",
                     smooth_value=None, effective_order=False)
   print('scores', bleu_scorer.corpus_score(sys_sents, refs_sents))
   print('fourth', bleu_scorer.corpus_score(sys_sents, refs_sents,).score)
   print(bleu_scorer.get_signature())
   print(bleu_scorer._aggregate_and_compute)
   # if sentence bleu then from easse!

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
   result = Result(score)
   print(score.estimate_ci)


   # # SIGNIFICANCE LEVELS for BLEU, chrf, TER
   # manual hacking of significance scores: 
   # #https://colab.research.google.com/drive/15nI8tPIhxBoLi4AjdnfoK_ERLU49xf5V?usp=sharing#scrollTo=zOqBe5DEdrix
   
   # !sacrebleu REFTXTREF.txt.sent -i ORHERFILEHERE.txt  -m bleu ter chrf --paired-bs
      
   # WER
   error = wer(human_ref, mt_hypothesis)
   print('word error rate (wer): ', error)

   
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
   reference = read_lines('./translations/sample_human_reference.txt')
   candidate = read_lines('./translations/sample_google_translate.txt')
   
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