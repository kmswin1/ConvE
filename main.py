# !coding=utf8
import json
import torch
import argparse
import os
from utils import batch_by_size, make_kg_vocab, graph_size, read_data
import time, datetime
from torch.utils.data import Dataset, DataLoader
from evaluation import ranking_and_hits

from model import ConvE, Complex

dir = os.getcwd() + '/data'

class KG_DataSet(Dataset):
    def __init__(self, file_path, kg_vocab):
        self.file_path = file_path
        self.kg_vocab = kg_vocab
        self.file = open(file_path, 'r')
        self.len = sum(1 for line in self.file.readlines())
        self.head = []
        self.rel = []
        self.tail = []
        for line in self.file.readlines():
            line = json.loads(line)
            self.head.append(kg_vocab.ent_id[line['e1']])
            self.rel.append(kg_vocab.ent_id[line['rel']])
            self.tails = []
            for t in line['e2_e1toe2'].split(' '):
                self.tails.append(kg_vocab.ent_id[t])
            self.tail.append(self.tails)
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return torch.LongTensor(self.head[index]), torch.LongTensor(self.rel[index]), torch.LongTensor(self.tail[index])


def main(args, model_path):
    print (os.getcwd())
    print ("start training ...")

    train_data = dir + '/train.json'
    #valid_data = dir + '/valid.json'
    test_data = dir + '/test.json'

    start = time.time()
    kg_vocab = make_kg_vocab(dir+'/e1rel_to_e2_full.json')
    print ("making vocab is done "+str(time.time()-start))
    n_ent, n_rel = graph_size(kg_vocab)

    #start = time.time()
    #train_data = read_data(os.path.join(dir, 'e1rel_toe2_train.json'), kg_vocab)
    #print ("making read_train data is done " + str(time.time()-start))

    #start = time.time()
    #train_data = read_data(os.path.join(dir, 'e1rel_toe2_test_ranking.json'), kg_vocab)
    #print ("making read_test data is done " + str(time.time()-start))

    #train_data = [torch.LongTensor(vec) for vec in train_data]


    model = ConvE(args, n_ent, n_rel)
    model.init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #if torch.cuda.device_count() > 1:
    #    model = torch.nn.DataParallel(model)
    #    criterion = torch.nn.BCELoss()
    model.cuda()
    print ('cuda : ' + str(torch.cuda.is_available()) + ' count : ' + str(torch.cuda.device_count()))

    params = [value.numel() for value in model.parameters()]
    print(params)
    print(sum(params))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    dataset = KG_DataSet(dir+'/e1rel_to_e2_train.json', kg_vocab)

    cnt = 0
    for epoch in range(args.epochs):
        print (epoch)
        epoch_loss = 0
        start = time.time()
        model.train()
        #h, r, t = train_data
        #n_train = h.size(0)
        #rand_idx = torch.randperm(n_train)
        #h = h[rand_idx].cuda()
        #r = r[rand_idx].cuda()
        tot = 0.0
        dataloader = torch.utils.data.DataLoader(dataset = dataset, num_workers=4, batch_size=args.batch_size, shuffle=True)


        for i, data in enumerate(dataloader):
            opt.zero_grad()
            batch_size = data.head.size(0)
            e2_multi = torch.empty(batch_size, n_ent, device=torch.device('cuda'))
            # label smoothing
            start = time.time()
            for i, t in enumerate(data.tail):
                e2_multi[i][t] = 1
            print ("e2_multi_time " +str(time.time()-start))
            e2_multi = ((1.0-args.label_smoothing)*e2_multi) + (1.0/e2_multi.shape[1])
            e2_multi = e2_multi.cuda()
            pred = model.forward(data.head, data.rel)
            loss = model.loss(pred, e2_multi)
            #loss = criterion(pred, e2_multi)
            loss.backward()
            opt.step()
            batch_loss = torch.sum(loss)
            epoch_loss += batch_loss
            tot += bh.size(0)
            print ('\r{:>10} progress {} loss: {}'.format('', tot/n_train, batch_loss), end='')
        epoch_loss /= batch_size
        print ('')
        end = time.time()
        time_used = end - start
        print ('one epoch time: {} minutes'.format(time_used/60))
        print ('{} epochs'.format(epoch))
        print ('epoch {} loss: {}'.format(epoch+1, epoch_loss))
        print ('saving to {0}'.format(model_path))
        torch.save(model.state_dict(), model_path)

        #model.eval()
        #with torch.no_grad():
        #    start = time.time()
        #    val_loss = ranking_and_hits(model, args.test_batch_size, test_data)
        #    end = time.time()
        #    print ('eval time used: {} minutes'.format((end - start)/60))

        #if epoch_loss < val_loss:
        #    cnt = 0
        #else:
        #    cnt += 1
        #    if cnt > 5:
        #        print ("Early stopping ...")
        #        break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KG completion for cruise contents data')
    parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training (default: 128)')
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