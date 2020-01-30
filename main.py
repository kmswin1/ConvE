# !coding=utf8
import json
import torch
import argparse
import os
from utils import make_kg_vocab, graph_size
import time, datetime
from torch.utils.data import Dataset, DataLoader
from evaluation import ranking_and_hits

from model import ConvE, Complex

dir = os.getcwd() + '/data'

class KG_DataSet(Dataset):
    def __init__(self, file_path, kg_vocab):
        self.kg_vocab = kg_vocab
        self.len = 0
        self.head = []
        self.rel = []
        self.tail = []
        with open(file_path) as f:
            for line in f:
                self.len += 1
                line = json.loads(line)
                self.head.append(self.kg_vocab.ent_id[line['e1']])
                self.rel.append(self.kg_vocab.rel_id[line['rel']])
                self.tail.append(line['e2_e1toe2'])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.head[idx], self.rel[idx], self.tail[idx]

class KG_EvalSet(Dataset):
    def __init__(self, file_path, kg_vocab):
        self.kg_vocab = kg_vocab
        self.len = 0
        self.head = []
        self.rel = []
        self.tail = []
        self.head2 = []
        self.rel_rev = []
        self.tail2 = []
        self.tails_len = []
        self.tails2_len  = []
        with open(file_path) as f:
            for line in f:
                self.len += 1
                line = json.loads(line)
                self.head.append(self.kg_vocab.ent_id[line['e1']])
                self.rel.append(self.kg_vocab.rel_id[line['rel']])
                self.tail.append(line['e2_e1toe2'])

                self.head2.append(self.kg_vocab.ent_id[line['e2']])
                self.rel_rev.append(self.kg_vocab.rel_id[line['rel_eval']])
                self.tail2.append(line['e2_e2toe1'])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.head[idx], self.rel[idx], self.tail[idx], self.head2[idx], self.rel_rev[idx], self.tail2[idx]

def main(args, model_path):
    print (os.getcwd())
    print ("start training ...")

    start = time.time()
    kg_vocab = make_kg_vocab(dir+'/e1rel_to_e2_full.json')
    print ("making vocab is done "+str(time.time()-start))
    n_ent, n_rel = graph_size(kg_vocab)


    model = ConvE(args, n_ent, n_rel)
    model.init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        criterion = torch.nn.BCELoss()
    model.cuda()
    print ('cuda : ' + str(torch.cuda.is_available()) + ' count : ' + str(torch.cuda.device_count()))

    params = [value.numel() for value in model.parameters()]
    print(params)
    print(sum(params))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    start = time.time()
    dataset = KG_DataSet(dir+'/e1rel_to_e2_train.json', kg_vocab)
    print ("making train dataset is done " + str(time.time()-start))
    start = time.time()
    evalset = KG_EvalSet(dir+'/e1rel_to_e2_ranking_test.json', kg_vocab)
    print ("making evalset is done " + str(time.time()-start))

    cnt = 0
    for epoch in range(args.epochs):
        print (epoch)
        epoch_loss = 0
        epoch_start = time.time()
        model.train()
        tot = 0.0
        dataloader = DataLoader(dataset=dataset, num_workers=args.num_worker, batch_size=args.batch_size, shuffle=True)
        n_train = dataset.__len__()

        for i, data in enumerate(dataloader):
            opt.zero_grad()
            head, rel, tail = data
            head = torch.LongTensor(head)
            rel = torch.LongTensor(rel)
            tail = tail.split('@@')
            tail = torch.LongTensor([kg_vocab[t] for t in tail])
            head = head.cuda()
            rel = rel.cuda()
            batch_size = head.size(0)
            epsilon = 1.0 / n_ent
            e2_multi = torch.full((batch_size, n_ent), epsilon)
            # label smoothing
            start = time.time()
            smoothed_value = 1 - args.label_smoothing
            for i, t in enumerate(tail):
                e2_multi[i][t] = smoothed_value + epsilon
            e2_multi = e2_multi.cuda()
            print ("e2_multi " + str(time.time()-start) + "\n")
            start = time.time()
            pred = model.forward(head, rel)
            #loss = model.loss(pred, e2_multi)
            loss = criterion(pred, e2_multi)
            loss.backward()
            opt.step()
            batch_loss = torch.sum(loss)
            print ("step " + str(time.time()-start) + "\n")
            epoch_loss += batch_loss
            tot += head.size(0)
            print ('\r{:>10} epoch {} progress {} loss: {}\n'.format('', epoch, tot/n_train, batch_loss), end='')
        epoch_loss /= batch_size
        print ('')
        end = time.time()
        time_used = end - epoch_start
        print ('one epoch time: {} minutes'.format(time_used/60))
        print ('{} epochs'.format(epoch))
        print ('epoch {} loss: {}'.format(epoch+1, epoch_loss))
        print ('saving to {0}'.format(model_path))
        torch.save(model.state_dict(), model_path)

        # TODO: calculate valid loss and develop early stopping

        model.eval()
        with torch.no_grad():
            start = time.time()
            ranking_and_hits(model, args, evalset, n_ent, epoch)
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
    parser.add_argument('--log-interval', type=int, default=100, help='how many batches to wait before logging training status')
    parser.add_argument('--data', type=str, default='webtoon', help='The kind of domain for training cruise data, default: person')
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