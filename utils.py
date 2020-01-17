import json
from itertools import count
from collections import namedtuple
from random import randint
from collections import defaultdict
import torch

def heads_tails(n_ent, train_data, valid_data=None, test_data=None):
    train_src, train_rel, train_dst = train_data
    if valid_data:
        valid_src, valid_rel, valid_dst = valid_data
    else:
        valid_src = valid_rel = valid_dst = []
    if test_data:
        test_src, test_rel, test_dst = test_data
    else:
        test_src = test_rel = test_dst = []
    all_src = train_src + valid_src + test_src
    all_rel = train_rel + valid_rel + test_rel
    all_dst = train_dst + valid_dst + test_dst
    heads = defaultdict(lambda: set())
    tails = defaultdict(lambda: set())
    for s, r, t in zip(all_src, all_rel, all_dst):
        tails[s, r].add(t)
        heads[t, r].add(s)
    heads_sp = {}
    tails_sp = {}
    for k in tails.keys():
        tails_sp[k] = torch.sparse.FloatTensor(torch.LongTensor([list(tails[k])]),
                                               torch.ones(len(tails[k])), torch.Size([n_ent]))
    for k in heads.keys():
        heads_sp[k] = torch.sparse.FloatTensor(torch.LongTensor([list(heads[k])]),
                                               torch.ones(len(heads[k])), torch.Size([n_ent]))
    return heads_sp, tails_sp


def inplace_shuffle(*lists):
    idx = []
    for i in range(len(lists[0])):
        idx.append(randint(0, i))
    for ls in lists:
        for i, item in enumerate(ls):
            j = idx[i]
            ls[i], ls[j] = ls[j], ls[i]


def batch_by_num(n_batch, *lists, n_sample=None):
    if n_sample is None:
        n_sample = len(lists[0])
    for i in range(n_batch):
        head = int(n_sample * i / n_batch)
        tail = int(n_sample * (i + 1) / n_batch)
        ret = [ls[head:tail] for ls in lists]
        if len(ret) > 1:
            yield ret
        else:
            yield ret[0]

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

def make_kg_vocab(*filenames):
    KBIndex = namedtuple('KBIndex', ['ent_list', 'rel_list', 'rel_reverse_list', 'ent_id', 'rel_id', 'rel_reverse_id'])
    ent_set = set()
    rel_set = set()
    rel_reverse = set()
    for filename in filenames:
        with open(filename) as f:
            for line in f:
                line = json.loads(line)
                e1 = line['src']
                rel = line['dstProperty']
                e2 = line['dst']
                rel_rev = rel + '_reverse'
                ent_set.add(e1)
                ent_set.add(e2)
                rel_set.add(rel)
                rel_reverse.add(rel_rev)
    ent_list = sorted(list(ent_set))
    rel_list = sorted(list(rel_set))
    rel_reverse_list = sorted(list(rel_reverse))
    ent_id = dict(zip(ent_list, count()))
    rel_id = dict(zip(rel_list, count()))
    rel_size = len(rel_id)
    rel_reverse_id = dict(zip(rel_reverse_list, count(rel_size)))
    return KBIndex(ent_list, rel_list, rel_reverse_list, ent_id, rel_id, rel_reverse_id)

def graph_size(vocab):
    return len(vocab.ent_id), len(vocab.rel_id)*2

def read_data(filename, kb_index):
    src = []
    rel = []
    dst = []
    with open(filename) as f:
        for line in f:
            line = json.loads(line)
            e1 = line['src']
            rel = line['dstProperty']
            e2 = line['dst']
            src.append(kb_index.ent_id[e1])
            rel.append(kb_index.rel_id[rel])
            dst.append(kb_index.ent_id[e2])
    return src, rel, dst

def read_reverse_data(filename, kb_index):
    src = []
    rel = []
    dst = []
    with open(filename) as f:
        for line in f:
            line = json.loads(line)
            e1 = line['src']
            rel = line['dstProperty']
            e2 = line['dst']
            rel_rev = rel + '_reverse'
            src.append(kb_index.ent_id[e2])
            rel.append(kb_index.rel_reverse_id[rel_rev])
            dst.append(kb_index.ent_id[e1])
    return src, rel, dst

def read_data_with_rel_reverse(filename, kb_index):
    src = []
    rel = []
    dst = []
    with open(filename) as f:
        for line in f:
            line = json.loads(line)
            e1 = line['src']
            rel = line['dstProperty']
            e2 = line['dst']
            rel_rev = rel + '_reverse'
            src.append(kb_index.ent_id[e1])
            rel.append(kb_index.rel_id[rel])
            dst.append(kb_index.ent_id[e2])
            src.append(kb_index.ent_id[e2])
            rel.append(kb_index.rel_reverse_id[rel_rev])
            dst.append(kb_index.ent_id[e1])
    return src, rel, dst