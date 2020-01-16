# !coding=utf8
import json
import torch
import pickle
import numpy as np
import argparse
import sys
import os
#from utils import heads_tails, inplace_shuffle, batch_by_num, batch_by_size, make_kg_vocab, graph_size, read_data, read_reverse_data, read_data_with_rel_reverse
import math
import pickle
import logging
import random
import time, datetime
from itertools import count
from collections import namedtuple
from evaluation import ranking_and_hits
from random import randint
from collections import defaultdict
from os.path import join
from logger import get_logger

logger = get_logger('train', True, True, 'training.txt')
logger.info('START TIME : {}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

from model import ConvE, Complex

dir = os.getcwd() + '/data'

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

def make_kg_vocab(*data):
    kg_vocab = namedtuple('kg_vocab', ['ent_list', 'rel_list', 'rel_rev_list', 'ent_id', 'rel_id', 'rel_rev_id'])
    ent_set = set()
    rel_set = set()
    rel_rev_set = set()
    for filename in data:
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
                rel_rev_set.add(rel_rev)
    ent_list = sorted(list(ent_set))
    rel_list = sorted(list(rel_set))
    rel_rev_list = sorted(list(rel_rev_set))
    ent_id = dict(zip(ent_list, count()))
    rel_id = dict(zip(rel_list, count()))
    len_rel = len(rel_id)
    rel_rev_id = dict(zip(rel_rev_set, count(len_rel)))
    return kg_vocab(ent_list, rel_list, rel_rev_list, ent_id, rel_id, rel_rev_id)

def graph_size(vocab):
    return len(vocab.ent_id), len(vocab.rel_id)*2

def read_data(filename, kg_vocab):
    src = []
    rel = []
    dst = []
    with open(filename) as f:
        for line in f:
            line = json.loads(line)
            h = line['src']
            r = line['dstProperty']
            t = line['dst']
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
            rel.append(kg_vocab.rel_id[r_revsers])
            dst.append(kg_vocab.ent_id[h])
    return src, rel, dst

def read_data_with_rel_reverse(filename, kg_vocab):
    src = []
    rel = []
    dst = []
    with open(filename) as f:
        for line in f:
            line = json.loads(line)
            h = line['src']
            r = line['dstProperty']
            t = line['dst']
            r_reverse = r + '_reverse'
            src.append(kg_vocab.ent_id[h])
            rel.append(kg_vocab.rel_id[r])
            dst.append(kg_vocab.ent_id[t])
            src.append(kg_vocab.ent_id[t])
            rel.append(kg_vocab.rel_rev_id[r_reverse])
            dst.append(kg_vocab.ent_id[h])
    return src, rel, dst

def main(args, model_path):
    print (os.getcwd())

    train_data = dir + '/train.json'
    valid_data = dir + '/valid.json'
    test_data = dir + '/test.json'

    kg_vocab = make_kg_vocab(train_data, valid_data, test_data)
    n_ent, n_rel = graph_size(kg_vocab)

    train_data_with_reverse = read_data_with_rel_reverse(os.path.join(dir, 'train.json'), kg_vocab)
    inplace_shuffle(*train_data_with_reverse)
    heads, tails = heads_tails(n_ent, train_data_with_reverse)

    train_data = read_data(os.path.join(dir, 'train.json'), kg_vocab)
    valid_data = read_data(os.path.join(dir, 'valid.json'), kg_vocab)
    test_data = read_data(os.path.join(dir, 'test.json'), kg_vocab)
    eval_h, eval_t = heads_tails(n_ent, train_data, valid_data, test_data)

    valid_data = [torch.LongTensor(vec) for vec in valid_data]
    test_data = [torch.LongTensor(vec) for vec in test_data]
    train_data_with_reverse = [torch.LongTensor(vec) for vec in train_data_with_reverse]



    model = ConvE(args, n_ent, n_rel)
    model.cuda() if torch.cuda.is_available() else model.cpu()
    print ('cuda : ' + str(torch.cuda.is_available()))

    model.init()
    params = [value.numel() for value in model.parameters()]
    print(params)
    print(sum(params))
    opt = torch.optim.Adam(model.parameters())

    for epoch in range(args.epochs):
        print (epoch)
        epoch_loss = 0
        start = time.time()
        model.train()
        h, r, t = train_data_with_reverse
        n_train = h.size(0)
        rand_idx = torch.randperm(n_train)
        h = h[rand_idx].cuda()
        r = r[rand_idx].cuda()
        tot = 0.0

        for bh, br in batch_by_num(args.batch_size, h, r):
            opt.zero_grad()
            batch_size = bh.size(0)
            e2_multi = torch.empty(batch_size, n_ent)
            # label smoothing
            for i, (head, rel) in enumerate(zip(bh, br)):
                head = head.item()
                rel = rel.item()
                e2_multi[i] = tails[head, rel].to_dense()
            e2_multi = ((1.0-args.label_smoothing)*e2_multi) + (1.0/e2_multi.shape[1])
            e2_multi = e2_multi.cuda()
            pred = model.forward(bh, br)
            loss = model.loss(pred, e2_multi)
            loss.backward()
            opt.step()
            batch_loss = torch.sum(loss)
            epoch_loss += batch_loss
            tot += bh.size(0)
            print('\r{:>10} progress {} loss: {}'.format('', tot/n_train, batch_loss), end='')
        logging.info('')
        end = time.time()
        time_used = end - start
        logging.info('one epoch time: {} minutes'.format(time_used/60))
        logging.info('{} epochs'.format(epoch))
        logging.info('epoch {} loss: {}'.format(epoch+1, epoch_loss))
        logging.info('saving to {0}'.format(model_path))
        torch.save(model.state_dict(), model_path)

        model.eval()
        with torch.no_grad():
            start = time.time()
            ranking_and_hits(model, args.batch_size, valid_data, eval_h, eval_t,'dev_evaluation')
            end = time.time()
            logging.info('eval time used: {} minutes'.format((end - start)/60))
            if epoch % 3 == 0:
                if epoch > 0:
                    ranking_and_hits(model, args.batch_size, test_data, eval_h, eval_t, 'test_evaluation')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KG completion for cruise contents data')
    parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=128, help='input batch size for testing/validation (default: 128)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate (default: 0.003)')
    parser.add_argument('--seed', type=int, default=17, metavar='S', help='random seed (default: 17)')
    parser.add_argument('--log-interval', type=int, default=100, help='how many batches to wait before logging training status')
    parser.add_argument('--data', type=str, default='entire', help='The kind of domain for training cruise data, default: person')
    parser.add_argument('--l2', type=float, default=0.0, help='Weight decay value to use in the optimizer. Default: 0.0')
    parser.add_argument('--model', type=str, default='conve', help='Choose from: {conve, distmult, complex}')
    parser.add_argument('--embedding-dim', type=int, default=200, help='The embedding dimension (1D). Default: 200')
    parser.add_argument('--embedding-shape1', type=int, default=20, help='The first dimension of the reshaped 2D embedding. The second dimension is infered. Default: 20')
    parser.add_argument('--hidden-drop', type=float, default=0.3, help='Dropout for the hidden layer. Default: 0.3.')
    parser.add_argument('--input-drop', type=float, default=0.2, help='Dropout for the input embeddings. Default: 0.2.')
    parser.add_argument('--feat-drop', type=float, default=0.2, help='Dropout for the convolutional features. Default: 0.2.')
    parser.add_argument('--lr-decay', type=float, default=0.995, help='Decay the learning rate by this factor every epoch. Default: 0.995')
    parser.add_argument('--loader-threads', type=int, default=4, help='How many loader threads to use for the batch loaders. Default: 4')
    parser.add_argument('--use-bias', action='store_true', help='Use a bias in the convolutional layer. Default: True')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing value to use. Default: 0.1')
    parser.add_argument('--hidden-size', type=int, default=9728, help='The side of the hidden layer. The required size changes with the size of the embeddings. Default: 9728 (embedding size 200).')
    parser.add_argument('--log-file', action='store', type=str)

    args = parser.parse_args()


    model_name = '{2}_{0}_{1}'.format(args.input_drop, args.hidden_drop, args.model)
    model_path = 'saved_models/{0}_{1}.model'.format(args.data, model_name)

    torch.manual_seed(args.seed)
    main(args, model_path)