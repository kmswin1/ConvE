# !coding=utf8
import json
import torch
import pickle
import numpy as np
import argparse
import sys
import os
from torch.utils.data import DataLoader, Dataset
import math
import pickle
import logging
import random
import time

from os.path import join

from model import ConvE, Complex

dir = os.getcwd()

def make_train_vocab(train_data):
    res = {}
    res['e1'] = set()
    res['rel'] = set()
    res['e2_e1toe2'] = set()
    with open(train_data) as f:
        for i, line in enumerate(f):
            line = json.loads(line)
            res['e1'][line['e1']] = i
            res['rel'][line['rel']] = i
            res['e2_e1toe2'][line['e2_e1toe2']] = i

    return res

def make_test_vocab(test_data):
    res = {}
    res['e1'] = set()
    res['rel'] = set()
    res['rel_reverse'] = set()
    res['e2_e1toe2'] = set()
    res['e2_e2toe1'] = set()
    with open(test_data) as f:
        for i, line in enumerate(f):
            line = json.loads(line)
            res['e1'][line['e1']] = i
            res['rel'][line['rel']] = i
            res['rel_eval'][line['rel_eval']] = i
            for meta in (line['e2_e1toe2'].split(' ')):
                res['e2_e1toe2'][meta] = i
            for meta in line(['e2_e2toe1'].split(' ')):
                res['e2_e2toe1'][meta] = i

    return res

def main(args):

    train_data = dir + '/data/e1rel_to_e2_train.json'
    valid_ranking_path = dir + '/data/e1rel_to_e2_ranking_valid.json'
    test_ranking_path = dir + '/data/e1rel_to_e2_ranking_test.json'

    train_vocab = make_train_vocab(train_data)
    valid_vocab = make_test_vocab(valid_ranking_path)
    test_vocab = make_test_vocab(test_ranking_path)

    print (train_vocab)

    train_batch = DataLoader(train_vocab, batch_size=args.batch_size, shuffle=True, num_workers=args.loader_threads)
    valid_batch = DataLoader(valid_vocab, batch_size=args.batch_size, shuffle=True, num_workers=args.loader_threads)
    test_batch = DataLoader(test_vocab, batch_size=args.batch_size, shuffle=True, num_workers=args.loader_threads)

    model = ConvE(args, len(train_vocab['e1']), len(train_vocab['rel']))
    model.cuda() if torch.cuda.is_available() else model.cpu()


if __name__ == '__main__':
    parser = argparse.AgumentParser(description='KG completion for cruise contents data')
    parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=128, help='input batch size for testing/validation (default: 128)')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate (default: 0.003)')
    parser.add_argument('--seed', type=int, default=17, metavar='S', help='random seed (default: 17)')
    parser.add_argument('--log-interval', type=int, default=100, help='how many batches to wait before logging training status')
    parser.add_argument('--data', type=str, default='person', help='The kind of domain for training cruise data, default: person')
    parser.add_argument('--ld', type=float, default=0.0, help='Weight decay value to use in the optimizer. Default: 0.0')
    parser.add_argument('--model', type=str, default='conve', help='Choose from: {conve, distmult, complex}')
    parser.add_argument('--embedding-dim', type=int, default=200, help='The embedding dimension (1D). Default: 200')
    parser.add_argument('--embedding-shape1', type=int, default=20, help='The first dimension of the reshaped 2D embedding. The second dimension is infered. Default: 20')
    parser.add_argument('--hidden-drop', type=float, default=0.3, help='Dropout for the hidden layer. Default: 0.3.')
    parser.add_argument('--input-drop', type=float, default=0.2, help='Dropout for the input embeddings. Default: 0.2.')
    parser.add_argument('--feat -drop', type=float, default=0.2, help='Dropout for the convolutional features. Default: 0.2.')
    parser.add_argument('--lr-decay', type=float, default=0.995, help='Decay the learning rate by this factor every epoch. Default: 0.995')
    parser.add_argument('--loader-threads', type=int, default=4, help='How many loader threads to use for the batch loaders. Default: 4')
    parser.add_argument('--preprocess', action='store_true', help='Preprocess the dataset. Needs to be executed only once. Default: 4')
    parser.add_argument('--resume', action='store_true', help='Resume a model.')
    parser.add_argument('--use-bias', action='store_true', help='Use a bias in the convolutional layer. Default: True')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing value to use. Default: 0.1')
    parser.add_argument('--hidden-size', type=int, default=9728, help='The side of the hidden layer. The required size changes with the size of the embeddings. Default: 9728 (embedding size 200).')
    parser.add_argument('--log-file', action='store', type=str)

    args = parser.parse_args()


    model_name = '{2}_{0}_{1}'.format(args.input_drop, args.hidden_drop, args.model)
    model_path = 'saved_models/{0}_{1}.model'.format(args.data, model_name)

    torch.manual_seed(args.seed)
    main(args, model_path)