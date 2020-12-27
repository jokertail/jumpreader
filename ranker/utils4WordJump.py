import json
import time
import logging

import numpy

from .utils import load_data as load_data_prototype
from .data import Dictionary
import torch
from tqdm import tqdm
import gc

logger = logging.getLogger(__name__)


def load_data(filename, mode=None):
    if mode:
        pass

    examples = []
    with open(filename) as f:
        for line in tqdm(f, desc="load examples"):
            l = json.loads(line)
            ex = {
                "question": l["question"],
                "ct": [i["str"] for i in l["context_true"]],
                "cf": [i["str"] for i in l["context_false"]]
            }
            examples.append(ex)

    return examples


def load_glove(filename, glove_dim=300):
    word_dict = Dictionary()
    word_emb = numpy.zeros((2, glove_dim), dtype=float)
    count = 0
    with open(filename) as f:
        for line in f:
            word, vec = line.split(' ', 1)
            word_dict.add(word)
            word_emb = numpy.vstack((word_emb, numpy.fromstring(vec, sep=' ')))
            count += 1

    return word_dict, word_emb


def get_sent_list(examples):
    pass
