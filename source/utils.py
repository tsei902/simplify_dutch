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


LANGUAGES = ['complex', 'simple']
PHASES = ['train', 'valid', 'test']
# SUBFOLDER = ['train', 'test']

def get_data_filepath(dataset, phase, type, i=None):
    suffix = ''
    print('line passed 1')
    if i is not None:
        suffix = f'.{i}'
        print('line passed 2', suffix)
    filename = f'{dataset}.{phase}.{type}{suffix}'
    print('this is the filename', filename)
    return DATASETS_DIR / dataset / filename


def get_dataset_dir(dataset):
    return DATASETS_DIR / dataset

def get_data_filepath(dataset, phase, language, i=None):
    suffix = ''  # Create suffix e.g. for multiple references
    if i is not None:
        suffix = f'.{i}'
    filename = f'{dataset}.{phase}.{language}{suffix}'
    return get_dataset_dir(dataset) / filename

def read_lines(filepath, n_lines=float('inf'), prop=1):
    return list(yield_lines(filepath, n_lines, prop))

def yield_lines(filepath, n_lines=float('inf'), prop=1):
    if prop < 1:
        assert n_lines == float('inf')
        n_lines = int(prop * count_lines(filepath))
    with open(filepath, 'r') as f:
        for i, l in enumerate(f):
            if i >= n_lines:
                break
            yield l.rstrip('\n')
            
def count_lines(filepath):
    n_lines = 0
    with Path(filepath).open() as f:
        for l in f:
            n_lines += 1
    return n_lines

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

        
def generate_hash(data):
    h = hashlib.new('md5')
    h.update(str(data).encode())
    return h.hexdigest()


def count_line(filepath):
    filepath = Path(filepath)
    line_count = 0
    with filepath.open("r", encoding="utf-8") as f:
        for line in f:
            line_count += 1
    return line_count


def read_lines(filepath):
    return [line.rstrip() for line in yield_lines(filepath)]