import os
import git
import sys
import time
import pandas as pd
from urllib.parse import urlparse
import logging
import shutil
import tempfile
from tqdm import tqdm
from pathlib import Path
from itertools import zip_longest
# from fcntl import flock, LOCK_EX, LOCK_UN
from urllib.request import urlretrieve
from contextlib import contextmanager, AbstractContextManager
import bz2
import gzip
import tarfile
import zipfile
import hashlib
import json

from paths import DATASETS_DIR # , CACHES_DIR

# -- fix path --
# from pathlib import Path
# import sys

# sys.path.append(str(Path(__file__).resolve().parent.parent))
# -- end fix path --

import pickle
import re
import sys
import time
from contextlib import contextmanager
from functools import lru_cache, wraps
from pathlib import Path
import hashlib

from paths import DUMPS_DIR, WIKILARGE_DATASET, get_temp_filepath, get_data_filepath
import json
from sacremoses import MosesDetokenizer, MosesTokenizer

# LANGUAGES = ['complex', 'simple']
# PHASES = ['train', 'valid', 'test']
# # SUBFOLDER = ['train', 'test']

# # CLEF
# def get_data_filepath(dataset, phase, type, i=None):
#     suffix = ''
#     print('line passed 1')
#     if i is not None:
#         suffix = f'.{i}'
#         print('line passed 2', suffix)
#     filename = f'{dataset}.{phase}.{type}{suffix}'
#     print('this is the filename', filename)
#     return DATASETS_DIR / dataset / filename

# # CLEF
# def get_dataset_dir(dataset):
#     return DATASETS_DIR / dataset

# #CLEF
# def read_lines(filepath, n_lines=float('inf'), prop=1):
#     return list(yield_lines(filepath, n_lines, prop))

# #CLEF
# def yield_lines(filepath, n_lines=float('inf'), prop=1):
#     if prop < 1:
#         assert n_lines == float('inf')
#         n_lines = int(prop * count_lines(filepath))
#     with open(filepath, 'r') as f:
#         for i, l in enumerate(f):
#             if i >= n_lines:
#                 break
#             yield l.rstrip('\n')

# # CLEF         
# def count_lines(filepath):
#     n_lines = 0
#     with Path(filepath).open() as f:
#         for l in f:
#             n_lines += 1
#     return n_lines

#CLEF
# def get_data_filepath(dataset, phase, language, i=None):
#     suffix = ''  # Create suffix e.g. for multiple references
#     if i is not None:
#         suffix = f'.{i}'
#     filename = f'{dataset}.{phase}.{language}{suffix}'
#     return get_dataset_dir(dataset) / filename

# CHENG SHEANG

@lru_cache(maxsize=1)
def get_tokenizer():
    return MosesTokenizer(lang='nl')

@lru_cache(maxsize=1)
def get_detokenizer():
    return MosesDetokenizer(lang='nl')

def tokenize(sentence):
    return get_tokenizer().tokenize(sentence)

def write_lines(lines, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w",encoding="utf-8") as fout:
        for line in lines:
            fout.write(line + '\n')


def read_lines(filepath):
    return [line.rstrip() for line in yield_lines(filepath)]


# # Returns file as list of lists
def read_file(folder_path):
    list = []
    with open(folder_path,  "r", encoding='utf8') as f:
        for line in f:
            line = line.rstrip('\n')
            line = [line]
            list.append(line)
    return list


def yield_lines(filepath):
    filepath = Path(filepath)
    with filepath.open('r', encoding="utf-8") as f: # is required in generation encoding and works with utf8
        # makes an issue at ?? 
        for line in f:
            yield line.rstrip()


def yield_sentence_pair_with_index(filepath1, filepath2):
    index = 0
    with Path(filepath1).open('r') as f1, Path(filepath2).open('r') as f2:
        for line1, line2 in zip(f1, f2):
            index += 1
            yield (line1.rstrip(), line2.rstrip(), index)


def yield_sentence_pair(filepath1, filepath2):
    with Path(filepath1).open('r', encoding="utf-8") as f1, Path(filepath2).open('r',encoding="utf-8") as f2:
        for line1, line2 in zip(f1, f2):
            yield line1.rstrip(), line2.rstrip()


def count_line(filepath):
    filepath = Path(filepath)
    line_count = 0
    with filepath.open("r", encoding='utf8') as f:
        for line in f:
            line_count += 1
    return line_count


def load_dump(filepath):
    return pickle.load(open(filepath, 'rb'))


def dump(obj, filepath):
    pickle.dump(obj, open(filepath, 'wb'))


def print_execution_time(func):
    @wraps(func)  # preserve name and doc of the function
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"Execution time({func.__name__}):{time.time() - start}")
        return result

    return wrapper


def generate_hash(data):
    h = hashlib.new('md5')
    h.update(str(data).encode())
    return h.hexdigest()


@contextmanager
def log_stdout(filepath, mute_stdout=False):
    '''Context manager to write both to stdout and to a file'''

    class MultipleStreamsWriter:
        def __init__(self, streams):
            self.streams = streams

        def write(self, message):
            for stream in self.streams:
                stream.write(message)

        def flush(self):
            for stream in self.streams:
                stream.flush()

    save_stdout = sys.stdout
    log_file = open(filepath, 'w')
    if mute_stdout:
        sys.stdout = MultipleStreamsWriter([log_file])  # Write to file only
    else:
        sys.stdout = MultipleStreamsWriter([save_stdout, log_file])  # Write to both stdout and file
    try:
        yield
    finally:
        sys.stdout = save_stdout
        log_file.close()


def log_params(filepath, kwargs):
    filepath = Path(filepath)
    kwargs_str = dict()
    for key in kwargs:
        kwargs_str[key] = str(kwargs[key])
    json.dump(kwargs_str, filepath.open('w'), indent=4)


def save_preprocessor(preprocessor):
    DUMPS_DIR.mkdir(parents=True, exist_ok=True)
    PREPROCESSOR_DUMP_FILE = DUMPS_DIR / 'preprocessor.pickle'
    dump(preprocessor, PREPROCESSOR_DUMP_FILE)


def load_preprocessor():
    PREPROCESSOR_DUMP_FILE = DUMPS_DIR / 'preprocessor.pickle'
    if PREPROCESSOR_DUMP_FILE.exists():
        return load_dump(PREPROCESSOR_DUMP_FILE)
    else:
        return None


def to_lrb_rrb(text):
    # TODO: Very basic
    text = re.sub(r'((^| ))\( ', r'\1-lrb- ', text)
    text = re.sub(r' \)((^| ))', r' -rrb-\1', text)
    return text


def apply_line_method_to_file(line_method, input_filepath):
    output_filepath = get_temp_filepath()
    with open(input_filepath, 'r') as input_file, open(output_filepath, 'w') as output_file:
        for line in input_file:
            transformed_line = line_method(line.rstrip('\n'))
            if transformed_line is not None:
                output_file.write(transformed_line + '\n')
    return output_filepath


def to_lrb_rrb_file(input_filepath):
    return apply_line_method_to_file(to_lrb_rrb, input_filepath)


def lowercase_file(filepath):
    return apply_line_method_to_file(lambda line: line.lower(), filepath)


def get_max_seq_length(dataset):
    # calculate sequence length
    max_length = 0
    for phase in ['train', 'valid']:
        complex_filepath = get_data_filepath(dataset, phase, 'orig')
        simple_filepath = get_data_filepath(dataset, phase, 'simp')
        for line in yield_lines(complex_filepath):
            l = len(line.split())
            if l > max_length: max_length = l

        for line in yield_lines(simple_filepath):
            l = len(line.split())
            if l > max_length: max_length = l
    return max_length


if __name__ == '__main__':
    # pass
    print(get_max_seq_length(WIKILARGE_DATASET))
