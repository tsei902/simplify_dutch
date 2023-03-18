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

from paths import DATASETS_DIR # , CACHES_DIR

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