import json
from itertools import count
from collections import namedtuple
from random import randint
from collections import defaultdict
import torch

#def heads_tails(n_ent, train_data, valid_data=None, test_data=None):


def inplace_shuffle(*lists):
    idx = []
    for i in range(len(lists[0])):
        idx.append(randint(0, i))
    for ls in lists:
        for i, item in enumerate(ls):
            j = idx[i]
            ls[i], ls[j] = ls[j], ls[i]


def batch_by_size(batch_size, *lists, n_sample=None):
    if n_sample is None:
        n_sample = len(lists[0])
    head = 0
    while head < n_sample:
        tail = min(n_sample, head + batch_size)
        ret = [ls[head:tail] for ls in lists]
        head += batch_size
        if len(ret) > 1:
            yield ret
        else:
            yield ret[0]

def make_kg_vocab(*data):
    kg_vocab = namedtuple('kg_vocab', ['ent_list', 'rel_list', 'ent_id', 'rel_id'])
    ent_set = set()
    rel_set = set()
    for filename in data:
        with open(filename) as f:
            for line in f:
                line = json.loads(line)
                e1 = line['e1']
                rel = line['rel']
                ent_set.add(e1)
                rel_set.add(rel)
    ent_list = sorted(list(ent_set))
    rel_list = sorted(list(rel_set))
    ent_id = dict(zip(ent_list, count()))
    rel_id = dict(zip(rel_list, count()))
    return kg_vocab(ent_list, rel_list, ent_id, rel_id)

def graph_size(vocab):
    return len(vocab.ent_id), len(vocab.rel_id)

def read_data(filename, kg_vocab):
    src = []
    rel = []
    dst = []
    with open(filename) as f:
        for line in f:
            line = json.loads(line)
            h = line['e1']
            r = line['rel']
            ts = line['e1_e1toe2'].split (' ')
            for t in ts:
                src.append(kg_vocab.ent_id[h])
                rel.append(kg_vocab.rel_id[r])
                dst.append(kg_vocab.ent_id[t])
    return src, rel, dst

def read_reverse_data(filename, kg_vocab):
    src = []
    rel = []
    dst = []
    with open(filename) as f:
        for line in f:
            line = json.loads(line)
            h = line['src']
            r = line['dstProperty']
            t = line['dst']
            r_revsers = r + '_reverse'
            src.append(kg_vocab.ent_id[t])
            rel.append(kg_vocab.rel_rev_id[r_revsers])
            dst.append(kg_vocab.ent_id[h])
    return src, rel, dst
