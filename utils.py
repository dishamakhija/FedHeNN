
import csv
from torchvision import datasets, transforms
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
import math
import random
from itertools import permutations
import numpy as np
import time
from torch.utils.data import DataLoader, Dataset
import copy
import math
from datetime import datetime
import json
import re


def read_data(train_file, test_file):
    '''parses data in given train and test data directories
    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    #train_files = os.listdir(train_data_dir)
    #train_files = [f for f in train_files if f.endswith('.json')]
    train_files = [train_file]
    for f in train_files:
        #file_path = os.path.join(train_data_dir,f)
        with open(f, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    #test_files = os.listdir(test_data_dir)
    #test_files = [f for f in test_files if f.endswith('.json')]
    test_files = [test_file]

    for f in test_files:
        #file_path = os.path.join(test_data_dir,f)
        with open(f, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    clients = list(train_data.keys())

    return clients, groups, train_data, test_data

ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
NUM_LETTERS = len(ALL_LETTERS)

def word_to_indices(word):
    '''returns a list of character indices
    Args:
        word: string
    Return:
        indices: int list with length len(word)
    '''
    indices = []
    for c in word:
        indices.append(ALL_LETTERS.find(c))
    return indices

def split_line(line):
    '''split given line/phrase into list of words
    Args:
        line: string representing phrase to be split

    Return:
        list of strings, with each string representing a word
    '''
    return re.findall(r"[\w']+|[.,!?;]", line)


def _word_to_index(word, indd):
    '''returns index of given word based on given lookup dictionary
    returns the length of the lookup dictionary if word not found
    Args:
        word: string
        indd: dictionary with string words as keys and int indices as values
    '''
    if word in indd:
        return indd[word]
    else:
        return len(indd)


def line_to_indices(line, word2id, max_words=25):
    '''converts given phrase into list of word indices

    if the phrase has more than max_words words, returns a list containing
    indices of the first max_words words
    if the phrase has less than max_words words, repeatedly appends integer
    representing unknown index to returned list until the list's length is
    max_words
    Args:
        line: string representing phrase/sequence of words
        word2id: dictionary with string words as keys and int indices as values
        max_words: maximum number of word indices in returned list
    Return:
        indl: list of word indices, one index for each word in phrase
    '''
    unk_id = len(word2id)
    line_list = split_line(line) # split phrase in words
    indl = [word2id[w] if w in word2id else unk_id for w in line_list[:max_words]]
    indl += [unk_id]*(max_words-len(indl))
    return indl


def bag_of_words(line, vocab):
    '''returns bag of words representation of given phrase using given vocab
    Args:
        line: string representing phrase to be parsed
        vocab: dictionary with words as keys and indices as values
    Return:
        integer list
    '''
    bag = [0]*len(vocab)
    words = split_line(line)
    for w in words:
        if w in vocab:
            bag[vocab[w]] += 1
    return bag

def get_word_emb_arr(path):
    with open(path, 'r') as inf:
        embs = json.load(inf)
    vocab = embs['vocab']
    word_emb_arr = np.array(embs['emba'])
    indd = {}
    for i in range(len(vocab)):
        indd[vocab[i]] = i
    vocab = {w: i for i, w in enumerate(embs['vocab'])}
    return word_emb_arr, indd, vocab

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def process_x(raw_x_batch, indd=None):
    """converts string of tokens to array of their indices in the embedding"""
    if indd is None:
        x_batch = [word_to_indices(word) for word in raw_x_batch]
        x_batch = np.array(x_batch).T
        return x_batch
    else:
        max_words = 25
        x_batch = raw_x_batch[4]
        x_batch = [line_to_indices(e, indd, max_words) for e in x_batch]
        x_batch = np.array(x_batch)
        return x_batch

def process_y(raw_y_batch, indd=None):
    """converts vector of labels to array whose rows are one-hot label vectors"""
    bs = raw_y_batch.shape[0]
    y = np.zeros((bs, 2), dtype=np.int8)
    for i in range(bs):
        y[i,raw_y_batch[i]] = int(1)
    return y
