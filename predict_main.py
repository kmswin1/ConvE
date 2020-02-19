# !coding=utf8
import json
import torch
import argparse
import os
import time, datetime
from pred_evaluation import ranking_and_hits
from model import ConvE, Complex
from utils import load_kg
from datasets import KG_EvalSet
from torch.utils.data import DataLoader

dir = os.getcwd() + '/data'

def main(args, model_path):
    print (os.getcwd())
    print ("start training ...")
    print (model_path)
    start = time.time()

    ent_str2id, ent_id2str, rel_str2id, rel_id2str = load_kg()
    print ("making vocab is done "+str(time.time()-start))
    n_ent, n_rel = len(ent_str2id), len(rel_str2id)

    model = ConvE(args, n_ent, n_rel)
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    print ('cuda : ' + str(torch.cuda.is_available()) + ' count : ' + str(torch.cuda.device_count()))

    params = [value.numel() for value in model.parameters()]
    print(params)
    print(sum(params))
    start = time.time()
    evalset = KG_EvalSet(dir+'/test_set.txt', args, n_ent)
    print ("making evalset is done " + str(time.time()-start))
    evalloader = DataLoader(dataset=evalset, num_workers=args.num_worker, batch_size=args.batch_size, shuffle=True)


    for epoch in range(args.test_batch_size):
        model.eval()
        with torch.no_grad():
            start = time.time()
            ranking_and_hits(model, args, evalloader, n_ent, ent_id2str, rel_id2str, epoch)
            end = time.time()
            print ('eval time used: {} minutes'.format((end - start)/60))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KG completion for cruise contents data')
    parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training (default: 128)')
    parser.add_argument('--num-worker', type=int, default=16, help='num_process of dataloader (default: 16)')
    parser.add_argument('--test-batch-size', type=int, default=128, help='input batch size for testing/validation (default: 128)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate (default: 0.003)')
    parser.add_argument('--seed', type=int, default=17, metavar='S', help='random seed (default: 17)')
    parser.add_argument('--data', type=str, default='cruise', help='The kind of domain for training cruise data, default: person')
    parser.add_argument('--l2', type=float, default=0.0, help='Weight decay value to use in the optimizer. Default: 0.0')
    parser.add_argument('--model', type=str, default='conve', help='Choose from: {conve, distmult, complex}')
    parser.add_argument('--embedding-dim', type=int, default=200, help='The embedding dimension (1D). Default: 200')
    parser.add_argument('--embedding-shape1', type=int, default=20, help='The first dimension of the reshaped 2D embedding. The second dimension is infered. Default: 20')
    parser.add_argument('--hidden-drop', type=float, default=0.3, help='Dropout for the hidden layer. Default: 0.3.')
    parser.add_argument('--input-drop', type=float, default=0.2, help='Dropout for the input embeddings. Default: 0.2.')
    parser.add_argument('--feat-drop', type=float, default=0.2, help='Dropout for the convolutional features. Default: 0.2.')
    parser.add_argument('--lr-decay', type=float, default=0.995, help='Decay the learning rate by this factor every epoch. Default: 0.995')
    parser.add_argument('--use-bias', action='store_true', help='Use a bias in the convolutional layer. Default: True')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing value to use. Default: 0.1')
    parser.add_argument('--multi-gpu', type=bool, default=False, help='choose the training using by multigpu')
    parser.add_argument('--feature-channel', type=int, default=32, help='The number of Feature map channels. Default: 32')

    args = parser.parse_args()


    model_name = '{2}_{0}_{1}'.format(args.input_drop, args.hidden_drop, args.model)
    model_path = 'saved_models/{0}_{1}.model'.format(args.data, model_name)

    torch.manual_seed(args.seed)
    main(args, model_path)