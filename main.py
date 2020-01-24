# !coding=utf8
import json
import torch
import argparse
import os
from utils import heads_tails, heads_tails_eval, inplace_shuffle, batch_by_size, make_kg_vocab, graph_size, read_data, read_reverse_data, read_data_with_rel_reverse
import time, datetime
from evaluation import ranking_and_hits

from model import ConvE, Complex

dir = os.getcwd() + '/data'

def main(args, model_path):
    print (os.getcwd())
    print ("start training ...")

    train_data = dir + '/train.json'
    #valid_data = dir + '/valid.json'
    test_data = dir + '/test.json'

    kg_vocab = make_kg_vocab(train_data, test_data)
    print ("making vocab is finish")
    n_ent, n_rel = graph_size(kg_vocab)

    train_data_with_reverse = read_data_with_rel_reverse(os.path.join(dir, 'train.json'), kg_vocab)
    train_data = read_data(os.path.join(dir, 'train.json'), kg_vocab)
    train_reverse = read_reverse_data(os.path.join(dir, 'train.json'), kg_vocab)
    inplace_shuffle(*train_data_with_reverse)
    heads, tails = heads_tails(n_ent, train_data_with_reverse)
    print ("making read train data and make train heads, tails is finish")


    #valid_data = read_data(os.path.join(dir, 'valid.json'), kg_vocab)
    test_data = read_data(os.path.join(dir, 'test.json'), kg_vocab)
    test_reverse = read_reverse_data(os.path.join(dir, 'test.json'), kg_vocab)
    eval_h, eval_t = heads_tails_eval(n_ent, train_data, train_reverse, test_data, test_reverse)
    print ("making read test data and make test heads, tails is finish")


    #valid_data = [torch.LongTensor(vec) for vec in valid_data]
    test_data = [torch.LongTensor(vec) for vec in test_data]
    train_data_with_reverse = [torch.LongTensor(vec) for vec in train_data_with_reverse]
    test_reverse = [torch.LongTensor(vec) for vec in test_reverse]


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

    cnt = 0
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

        for bh, br in batch_by_size(args.batch_size, h, r):
            opt.zero_grad()
            batch_size = bh.size(0)
            e2_multi = torch.empty(batch_size, n_ent, device=torch.device('cuda'))
            # label smoothing
            for i, (head, rel) in enumerate(zip(bh, br)):
                head = head.item()
                rel = rel.item()
                e2_multi[i] = tails[head, rel].to_dense()
            e2_multi = ((1.0-args.label_smoothing)*e2_multi) + (1.0/e2_multi.shape[1])
            e2_multi = e2_multi.cuda()
            pred = model.forward(bh, br)
            #loss = model.loss(pred, e2_multi)
            loss = criterion(pred, e2_multi)
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

        model.eval()
        with torch.no_grad():
            start = time.time()
            val_loss = ranking_and_hits(model, args.test_batch_size, test_data, test_reverse, eval_h, eval_t,'dev_evaluation', epoch)
            end = time.time()
            print ('eval time used: {} minutes'.format((end - start)/60))

        if epoch_loss < val_loss:
            cnt = 0
        else:
            cnt += 1
            if cnt > 5:
                print ("Early stopping ...")
                break


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