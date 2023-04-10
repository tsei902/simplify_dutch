from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from source import utils
from functools import lru_cache
from multiprocessing import Pool
from string import punctuation
import multiprocessing
import Levenshtein
import numpy as np
import spacy
import os
import nltk
import tarfile
import zipfile
import urllib
import pickle
from tqdm import tqdm
import shutil
import gensim
import time
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
import re
from paths import DUMPS_DIR, ASSET_DATASET,  PHASES, get_data_filepath, PROCESSED_DATA_DIR, \
    DATASETS_DIR, WIKILARGE_DATASET, WORD_EMBEDDINGS_NAME 
    # WORD_FREQUENCY_FILEPATH 
from utils import tokenize, yield_lines, load_dump, dump, write_lines, count_line, \
    print_execution_time, save_preprocessor, yield_sentence_pair

stopwords = set(stopwords.words('dutch'))
from compound_split import doc_split

def round(val):
    return '%.2f' % val

def safe_division(a, b):
    return a / b if b else 0

# def tokenize(sentence):
#     return sentence.split()

@lru_cache(maxsize=1024)
def is_punctuation(word):
    return ''.join([char for char in word if char not in punctuation]) == ''

@lru_cache(maxsize=128)
def remove_punctuation(text):
    return ' '.join([word for word in tokenize(text) if not is_punctuation(word)])

def remove_stopwords(text):
    return ' '.join([w for w in tokenize(text) if w.lower() not in stopwords])

@lru_cache(maxsize=1024)
def get_dependency_tree_depth(sentence):
    def tree_height(node):
        if len(list(node.children)) == 0:
            return 0
        return 1 + max(tree_height(child) for child in node.children)

    tree_depths = [tree_height(spacy_sentence.root) for spacy_sentence in spacy_process(sentence).sents]
    if len(tree_depths) == 0:
        return 0
    return max(tree_depths)

@lru_cache(maxsize=1)
def get_spacy_model():
    model = 'nl_core_news_sm'  # from spacy, Dutch pipeline optimized for CPU. Components: tok2vec, morphologizer, tagger, parser, lemmatizer (trainable_lemmatizer), senter, ner.
    if not spacy.util.is_package(model):
        spacy.cli.download(model)
        spacy.cli.link(model, model, force=True, model_path=spacy.util.get_package_path(model))
    return spacy.load(model)

@lru_cache(maxsize=10 ** 6)
def spacy_process(text):
    return get_spacy_model()(str(text))

@lru_cache(maxsize=1)
def get_word2rank(vocab_size=np.inf):
    model_filepath = DUMPS_DIR / f"{WORD_EMBEDDINGS_NAME}.pk"  # should be CoNNL 17 model!
    if model_filepath.exists():
        return load_dump(model_filepath)
    else:
        print("Downloading alterantive coostco dutch embeddings ...") # pretrained vectors
        download_twitter_embeddings(model_name='coostco', dest_dir=str(DUMPS_DIR))
        print("Preprocessing word2rank...")
        DUMPS_DIR.mkdir(parents=True, exist_ok=True)
        WORD_EMBEDDINGS_PATH = DUMPS_DIR / f'{WORD_EMBEDDINGS_NAME}.bin'
        model = load_word_embeddings(WORD_EMBEDDINGS_PATH) # returns index_to_key
        # store into file
        lines_generator = model 
        word2rank = {}
        print('vocab_size', vocab_size)
        for i, line in enumerate(lines_generator):
            if i >= vocab_size: break # its not vocab size any more but  # len(model.key_to_index)
            word = line.split(',')[0]
            word2rank[word] = i
        pickle.dump(word2rank, open(model_filepath, 'wb'))
        # txt_file = DUMPS_DIR / f'{WORD_EMBEDDINGS_NAME}.txt'
        # zip_file = DUMPS_DIR / f'{WORD_EMBEDDINGS_NAME}.zip'
    return word2rank        
    
def load_word_embeddings(filepath):
    model = gensim.models.KeyedVectors.load_word2vec_format(filepath, binary=True) # '../resources/DUMPS/model.bin'
    model_indexes = model.index_to_key
    return model_indexes   
    
def download_twitter_embeddings(model_name, dest_dir): # pretrained rankings
    url = ''
    if model_name == 'coosto_model':
        url = 'https://github.com/coosto/dutch-word-embeddings/releases/download/v1.0/model.bin'
    file_path = download_url(url, dest_dir)
    out_filepath = Path(file_path)
    out_filepath = out_filepath.parent / f'{out_filepath.stem}.txt'
    # print(out_filepath, out_filepath.exists())
    if not out_filepath.exists():
        print("Extracting: ", Path(file_path).name)
        unzip(file_path, dest_dir) 
        
def download_url(self, url, output_path):
        name = url.split('/')[-1]
        file_path = f'{output_path}/{name}'
        if not Path(file_path).exists():
            with tqdm(unit='B', unit_scale=True, leave=True, miniters=1,desc=name) as t:  # all optional kwargs
                urllib.request.urlretrieve(url, filename=file_path, reporthook=self._download_report_hook(t), data=None)
        return file_path   
    
def unzip(self, file_path, dest_dir=None):
    if dest_dir is None:
        dest_dir = os.path.dirname(file_path)
    if file_path.endswith('.zip'):
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(dest_dir)
    elif file_path.endswith("tar.gz") or file_path.endswith("tgz"):
        tar = tarfile.open(file_path, "r:gz")
        tar.extractall(dest_dir)
        tar.close()
    elif file_path.endswith("tar"):
        tar = tarfile.open(file_path, "r:")
        tar.extractall(dest_dir)
        tar.close()

@lru_cache(maxsize=10000)
def get_normalized_rank(word):
    max = len(get_word2rank())
    rank = get_word2rank().get(word, max)
    return np.log(1 + rank) / np.log(1 + max)
    # return np.log(1 + rank)
    
@lru_cache(maxsize=2048)
def get_complexity_score2(sentence):
    words = tokenize(remove_stopwords(remove_punctuation(sentence)))
    words = [word for word in words if word in get_word2rank()]  # remove unknown words
    if len(words) == 0:
        return 1.0
    return np.array([get_normalized_rank(word) for word in words]).mean()

# @lru_cache(maxsize=1)
# def get_word_frequency():
#     model_filepath = DUMPS_DIR / f'{WORD_FREQUENCY_FILEPATH.stem}.pk'
#     if model_filepath.exists():
#         return load_dump(model_filepath)
#     else:
#         DUMPS_DIR.mkdir(parents=True, exist_ok=True) 
#         word_freq = {}
#         for line in yield_lines(WORD_FREQUENCY_FILEPATH):
#             chunks = line.split(' ')
#             word = chunks[0]
#             freq = int(chunks[1])
#             word_freq[word] = freq
#         dump(word_freq, model_filepath)
#         return word_freq

# @lru_cache(maxsize=10000)
# def get_normalized_frequency(word):
#     max = 153141437 # the 153141437, the max frequency
#     freq = get_word_frequency().get(word, 0)
#     return 1.0 - np.log(1 + freq) / np.log(1 + max)


# @lru_cache(maxsize=2048)
# def get_complexity_score(sentence):
#     # words = tokenize(remove_stopwords(remove_punctuation(sentence)))
#     words = tokenize(remove_punctuation(sentence))
#     words = [word for word in words if word in get_word2rank()]  # remove unknown words
#     if len(words) == 0:
#         return 1.0
    
#     return np.array([get_normalized_frequency(word.lower()) for word in words]).mean()

def download_requirements():
    get_spacy_model()
    get_word2rank()

class RatioFeature:
    def __init__(self, feature_extractor, target_ratio=0.8):
        self.feature_extractor = feature_extractor
        self.target_ratio = target_ratio

    def encode_sentence(self, sentence):
        return f'{self.name}_{self.target_ratio}'

    def encode_sentence_pair(self, complex_sentence, simple_sentence):
        return f'{self.name}_{self.feature_extractor(complex_sentence, simple_sentence)}', simple_sentence

    def decode_sentence(self, encoded_sentence):
        return encoded_sentence

    @property
    def name(self):
        class_name = self.__class__.__name__.replace('RatioFeature', '')
        name = ""
        for word in re.findall('[A-Z][^A-Z]*', class_name):
            if word: name += word[0]
        if not name: name = class_name
        return name

class WordRatioFeature(RatioFeature):
    def __init__(self, *args, **kwargs):
        super().__init__(self.get_word_length_ratio, *args, **kwargs)

    def get_word_length_ratio(self, complex_sentence, simple_sentence):
        return round(safe_division(len(tokenize(simple_sentence)), len(tokenize(complex_sentence))))


class CharRatioFeature(RatioFeature):
    def __init__(self, *args, **kwargs):
        super().__init__(self.get_char_length_ratio, *args, **kwargs)

    def get_char_length_ratio(self, complex_sentence, simple_sentence):
        return round(safe_division(len(simple_sentence), len(complex_sentence)))


class LevenshteinRatioFeature(RatioFeature):
    def __init__(self, *args, **kwargs):
        super().__init__(self.get_levenshtein_ratio, *args, **kwargs)

    def get_levenshtein_ratio(self, complex_sentence, simple_sentence):
        # old return round(Levenshtein.ratio(complex_sentence, simple_sentence))
        complex_sentence = tokenize(complex_sentence)
        simple_sentence = tokenize(simple_sentence)
        return round(Levenshtein.seqratio(complex_sentence, simple_sentence))
    
class WordRankRatioFeature(RatioFeature):
    def __init__(self, *args, **kwargs):
        super().__init__(self.get_word_rank_ratio, *args, **kwargs)

    def get_word_rank_ratio(self, complex_sentence, simple_sentence):
        score = round(min(safe_division(self.get_lexical_complexity_score(simple_sentence),
                                       self.get_lexical_complexity_score(complex_sentence)), 2))
        print('score', score)
        return score

    def get_lexical_complexity_score(self, sentence):
        # print('enter lexical loop')
        words = tokenize(remove_stopwords(remove_punctuation(sentence)))
        # print('sentence "tokenization" into individal words', words)
        words = doc_split.maximal_split(words)
        # print('result nach max split', words)
        words = [word for word in words if word in get_word2rank()]
        # print('words here is the check if the word exists?', words)
        # print('still in lexical loop')
        if len(words) == 0:
            return np.log(1 + len(get_word2rank()))
        score =  np.quantile([self.get_rank(word) for word in words], 0.75)
        # print('score for each word', score)
        # print('lexical compexity score', score)
        return score

    @lru_cache(maxsize=5000)
    def get_rank(self, word):
        rank = get_word2rank().get(word, len(get_word2rank()))
        # print('rank of word from word2rank - glove ', rank)
        ranker = np.log(1 + rank)
        # print('ranker: ', ranker)
        return ranker

class DependencyTreeDepthRatioFeature(RatioFeature):
    def __init__(self, *args, **kwargs):
        super().__init__(self.get_dependency_tree_depth_ratio, *args, **kwargs)

    def get_dependency_tree_depth_ratio(self, complex_sentence, simple_sentence):
        return round(
            safe_division(self.get_dependency_tree_depth(simple_sentence),
                          self.get_dependency_tree_depth(complex_sentence)))

    @lru_cache(maxsize=1024)
    def get_dependency_tree_depth(self, sentence):
        def get_subtree_depth(node):
            if len(list(node.children)) == 0:
                return 0
            return 1 + max([get_subtree_depth(child) for child in node.children])

        tree_depths = [get_subtree_depth(spacy_sentence.root) for spacy_sentence in self.spacy_process(sentence).sents]
        if len(tree_depths) == 0:
            return 0
        return max(tree_depths)

    @lru_cache(maxsize=10 ** 6)
    def spacy_process(self, text):
        return get_spacy_model()(text)


class Preprocessor:
    def __init__(self, features_kwargs=None):
        super().__init__()

        self.features = self.get_features(features_kwargs)
        if features_kwargs:
            self.hash = utils.generate_hash(str(features_kwargs).encode())
        else:
            self.hash = "no_feature"

    def get_class(self, class_name, *args, **kwargs):
        return globals()[class_name](*args, **kwargs)

    def get_features(self, feature_kwargs):
        features = []
        for feature_name, kwargs in feature_kwargs.items():
            features.append(self.get_class(feature_name, **kwargs))
        return features

    def encode_sentence(self, sentence):
        if self.features:
            line = ''
            for feature in self.features:
                line += feature.encode_sentence(sentence) + ' '
            line += ' ' + sentence
            # print('featured sentence', line)
            return line.rstrip()
        else:
            return sentence

    def encode_sentence_pair(self, complex_sentence, simple_sentence):
        # print(complex_sentence)
        if self.features:
            line = ''
            for feature in self.features:
                # startTime = timeit.default_timer()
                # print(feature)
                processed_complex, _ = feature.encode_sentence_pair(complex_sentence, simple_sentence)
                line += processed_complex + ' '
                # print('featured sentence', line)
                # print(feature, timeit.default_timer() - startTime)
            line += ' ' + complex_sentence
            return line.rstrip()

        else:
            return complex_sentence

    def decode_sentence(self, encoded_sentence):
        for feature in self.features:
            decoded_sentence = feature.decode_sentence(encoded_sentence)
        return decoded_sentence

    def encode_file(self, input_filepath, output_filepath):
        with open(output_filepath, 'w') as f:
            for line in yield_lines(input_filepath):
                f.write(self.encode_sentence(line) + '\n')

    def decode_file(self, input_filepath, output_filepath):
        with open(output_filepath, 'w') as f:
            for line in yield_lines(input_filepath):
                f.write(self.decode_sentence(line) + '\n')

    def process_encode_sentence_pair(self, sentences):
        print(f"{sentences[2]}/{self.line_count}", sentences[0])  # sentence[0] index
        return (self.encode_sentence_pair(sentences[0], sentences[1]))

    def pool_encode_sentence_pair(self, args):
        # print(f"{processed_line_count}/{self.line_count}")
        complex_sent, simple_sent, queue = args
        queue.put(1)
        return self.encode_sentence_pair(complex_sent, simple_sent)

    @print_execution_time
    def encode_file_pair(self, complex_filepath, simple_filepath):
        print(f"Preprocessing file: {complex_filepath}")
        processed_complex_sentences = []
        self.line_count = count_line(simple_filepath)

        nb_cores = multiprocessing.cpu_count()
        manager = multiprocessing.Manager()
        queue = manager.Queue()

        pool = Pool(processes=nb_cores)
        args = [(complex_sent, simple_sent, queue) for complex_sent, simple_sent in
                yield_sentence_pair(complex_filepath, simple_filepath)]
        res = pool.map_async(self.pool_encode_sentence_pair, args)
        while not res.ready():
            # remaining = res._number_left * res._chunksize
            size = queue.qsize()
            print(f"Preprocessing: {size} / {self.line_count}")
            time.sleep(0.5)
        encoded_sentences = res.get()
        pool.close()
        pool.join()
        # pool.terminate()
        # i = 0
        # for complex_sentence, simple_sentence in yield_sentence_pair(complex_filepath, simple_filepath):
        # # print(complex_sentence)
        #     processed_complex_sentence = self.encode_sentence_pair(complex_sentence, simple_sentence)
        #     i +=1
        #     print(f"{i}/{self.line_count}", processed_complex_sentence)
        # processed_complex_sentences.append(encoded_complex)

        return encoded_sentences

    def get_preprocessed_filepath(self, dataset, phase, type):
        filename = f'{dataset}.{phase}.{type}'
        return self.preprocessed_data_dir / filename

    def preprocess_dataset(self, dataset):
        # download_requirements()
        # print('self.hash', self.hash)
        print('dataset', dataset)
        self.preprocessed_data_dir = PROCESSED_DATA_DIR /  dataset #self.hash /
        self.preprocessed_data_dir.mkdir(parents=True, exist_ok=True)
        save_preprocessor(self)
        print(f'Preprocessing dataset: {dataset}')

        for phase in PHASES:
            # for phase in ["train", "valid"]: 
            complex_filepath = get_data_filepath(dataset, phase, 'orig')
            simple_filepath = get_data_filepath(dataset, phase, 'simp')

            complex_output_filepath = self.preprocessed_data_dir / complex_filepath.name
            simple_output_filepath = self.preprocessed_data_dir / simple_filepath.name
            if complex_output_filepath.exists() and simple_output_filepath.exists():
                continue

            print(f'Prepocessing files: {complex_filepath.name} {simple_filepath.name}')
            processed_complex_sentences = self.encode_file_pair(complex_filepath, simple_filepath)

            write_lines(processed_complex_sentences, complex_output_filepath)
            shutil.copy(simple_filepath, simple_output_filepath)

        print(f'Preprocessing dataset "{dataset}" is finished.')
        return self.preprocessed_data_dir


if __name__ == '__main__':

    features_kwargs = {
        'WordRatioFeature': {'target_ratio': 0.8},
        'CharRatioFeature': {'target_ratio': 0.8},
        'LevenshteinRatioFeature': {'target_ratio': 0.8},
        'WordRankRatioFeature': {'target_ratio': 0.8},
        'DependencyTreeDepthRatioFeature': {'target_ratio': 0.8}
    }
    # features_kwargs = {}
    preprocessor = Preprocessor(features_kwargs)
    # preprocessor.preprocess_dataset(ASSET_DATASET)
    preprocessor.preprocess_dataset(WIKILARGE_DATASET)
    # preprocessor.preprocess_dataset(NEWSELA_DATASET)
