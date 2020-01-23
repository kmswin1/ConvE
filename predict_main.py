# !coding=utf8
import json
import torch
import argparse
import os
from utils import heads_tails, inplace_shuffle, batch_by_num, batch_by_size, make_kg_vocab, graph_size, read_data, read_reverse_data, read_data_with_rel_reverse
import time, datetime
from pred_evaluation import ranking_and_hits
from model import ConvE, Complex
from utils import heads_tails_eval

dir = os.getcwd() + '/data'


def main(args, model_path):
    print (os.getcwd())
    print ('start prediction ...')

    train_data = dir + '/train.json'
    valid_data = dir + '/valid.json'
    test_data = dir + '/test.json'

    kg_vocab = make_kg_vocab(train_data, test_data)
    n_ent, n_rel = graph_size(kg_vocab)

    train_data_with_reverse = read_data_with_rel_reverse(os.path.join(dir, 'train.json'), kg_vocab)
    inplace_shuffle(*train_data_with_reverse)

    train_data = read_data(os.path.join(dir, 'train.json'), kg_vocab)
    train_reverse = read_reverse_data(os.path.join(dir, 'train.json'), kg_vocab)
    test_data = read_data(os.path.join(dir, 'test.json'), kg_vocab)
    test_reverse = read_reverse_data(os.path.join(dir, 'test.json'), kg_vocab)
    eval_h, eval_t = heads_tails_eval(n_ent, train_data, train_reverse, test_data, test_reverse)

    test_data = [torch.LongTensor(vec) for vec in test_data]
    test_reverse = [torch.LongTensor(vec) for vec in test_reverse]

    model = ConvE(args, n_ent, n_rel)
    model.cuda() if torch.cuda.is_available() else model.cpu()
    print ('cuda : ' + str(torch.cuda.is_available()))
    model.load_state_dict(torch.load(dir+args.model_path))
    print (model)

    model.eval()
    with torch.no_grad():
        start = time.time()
        ranking_and_hits(model, args.batch_size, test_data, test_reverse, eval_h, eval_t,'prediction', kg_vocab)
        end = time.time()
        print ('eval time used: {} minutes'.format((end - start)/60))


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
    parser.add_argument('--model-path', type=str, default='/saved_models/webtoon_conve_0.2_0.3.model', help='define your model to evaluate')

    args = parser.parse_args()


    model_name = '{2}_{0}_{1}'.format(args.input_drop, args.hidden_drop, args.model)
    model_path = 'saved_models/{0}_{1}.model'.format(args.data, model_name)

    torch.manual_seed(args.seed)
    main(args, model_path)