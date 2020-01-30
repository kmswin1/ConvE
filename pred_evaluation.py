import torch
import numpy as np
import datetime, os, pickle
from torch.utils.data import DataLoader
import time
#timer = CUDATimer()
dir = os.getcwd()
def ranking_and_hits(model, args, testset, n_ent, kg_vocab, epoch):
    dataloader = DataLoader(dataset=testset, num_workers=4, batch_size=args.batch_size, shuffle=True)
    hits_left = []
    hits_right = []
    hits = []
    ranks = []
    ranks_left = []
    ranks_right = []

    for i in range(10):
        hits_left.append([])
        hits_right.append([])
        hits.append([])

    for i, data in enumerate(dataloader):
        head, rel, tail, head2, rel_rev, tail2 = data
        head = torch.LongTensor(head)
        rel = torch.LongTensor(rel)
        tails = []
        for meta in tail:
            meta = meta.split('@@')
            temp = []
            for t in meta:
                temp.append(kg_vocab.ent_id[t])
            tails.append(temp)
        tails = [torch.LongTensor(vec) for vec in tails]
        head2 = torch.LongTensor(head2)
        rel_rev = torch.LongTensor(rel_rev)
        tails2 = []
        for meta in tail:
            meta = meta.split('@@')
            temp = []
            for t in meta:
                temp.append(kg_vocab.ent_id[t])
            tails.append(temp)
        tails2 = [torch.LongTensor(vec) for vec in tails2]
        head = head.cuda()
        rel = rel.cuda()
        head2 = head2.cuda()
        rel_rev = rel_rev.cuda()
        batch_size = head.size(0)

        e2_multi1 = torch.zeros(batch_size, n_ent, dtype=torch.int64)
        e2_multi2 = torch.zeros(batch_size, n_ent, dtype=torch.int64)
        for i, (t,t_r) in enumerate(zip(tails, tails2)):
            e2_multi1[i][t] = 1
            e2_multi2[i][t_r] = 1
        e2_multi1 = e2_multi1.cuda()
        e2_multi2 = e2_multi2.cuda()
        pred1 = model.forward(head, rel)
        pred2 = model.forward(head2, rel_rev)

        for i in range(batch_size):
            # save the prediction that is relevant
            target_value1 = pred1[i][head2[i]].item()
            target_value2 = pred2[i][head[i]].item()
            # zero all known cases (this are not interesting)
            # this corresponds to the filtered setting
            pred1[i][e2_multi1[i]] = 0.0
            pred2[i][e2_multi2[i]] = 0.0
            # write base the saved values
            pred1[i][head2[i]] = target_value1
            pred2[i][head[i]] = target_value2

        # sort and rank
        max_values, argsort1 = torch.sort(pred1, 1, descending=True)
        max_values, argsort2 = torch.sort(pred2, 1, descending=True)
        for i in range(batch_size):
            # find the rank of the target entities
            find_target1 = argsort1[i] == head2[i]
            find_target2 = argsort2[i] == head[i]
            rank1 = torch.nonzero(find_target1)[0, 0].item() + 1
            rank2 = torch.nonzero(find_target2)[0, 0].item() + 1
            # rank+1, since the lowest rank is rank 1 not rank 0
            ranks.append(rank1+1)
            ranks_left.append(rank1)
            ranks.append(rank2+1)
            ranks_right.append(rank2)

            # this could be done more elegantly, but here you go
            hits[9].append(int(rank1<=10))
            hits[9].append(int(rank2<=10))
            hits_left[9].append((int(rank1<=10)))
            hits_right[9].append((int(rank2<=10)))
            # for hits_level in range(10):
            #     if rank1 <= hits_level:
            #         hits[hits_level].append(1.0)
            #         hits_left[hits_level].append(1.0)
            #     else:
            #         hits[hits_level].append(0.0)
            #         hits_left[hits_level].append(0.0)
            #
            #     if rank2 <= hits_level:
            #         hits[hits_level].append(1.0)
            #         hits_right[hits_level].append(1.0)
            #     else:
            #         hits[hits_level].append(0.0)
            #         hits_right[hits_level].append(0.0)
            if rank1 == 1:
                with open(dir + '/log_file/hit.txt', 'a') as f:
                    f.write(kg_vocab.ent_list[head[i].item()]+"\n")
                    f.write(kg_vocab.rel_list[rel[i].item()]+"\n")
                    f.write(kg_vocab.ent_list[argsort1[i][0].item()]+"\n")
                    f.close()
            elif rank1 > 1:
                with open(dir + '/log_file/nohit.txt', 'a') as f:
                    f.write(kg_vocab.ent_list[head[i].item()]+"\n")
                    f.write(kg_vocab.rel_list[rel[i].item()]+"\n")
                    f.write(kg_vocab.ent_list[argsort1[i][0].item()]+"\n")
                    f.close()
            if rank2 == 1:
                with open(dir + '/log_file/hit.txt', 'a') as f:
                    f.write(kg_vocab.ent_list[head2[i].item()]+"\n")
                    f.write(kg_vocab.rel_rev_list[rel_rev[i].item()]+"\n")
                    f.write(kg_vocab.ent_list[argsort2[i][0].item()]+"\n")
                    f.close()
            elif rank2 > 1:
                with open(dir + '/log_file/nohit.txt', 'a') as f:
                    f.write(kg_vocab.ent_list[head2[i].item()]+"\n")
                    f.write(kg_vocab.rel_rev_list[rel_rev[i].item()]+"\n")
                    f.write(kg_vocab.ent_list[argsort2[i][0].item()]+"\n")
                    f.close()


    print('Hits tail @{0}: {1}'.format(10, np.mean(hits_left[9])))
    print('Hits head @{0}: {1}'.format(10, np.mean(hits_right[9])))
    print('Hits @{0}: {1}'.format(10, np.mean(hits[9])))
    print('Mean rank tail: {0}'.format(np.mean(ranks_left)))
    print('Mean rank head: {0}'.format(np.mean(ranks_right)))
    print('Mean rank: {0}'.format(np.mean(ranks_left+ranks_right)))
    print('Mean reciprocal rank tail: {0}'.format(np.mean(1./np.array(ranks_left))))
    print('Mean reciprocal rank head: {0}'.format(np.mean(1./np.array(ranks_right))))
    print('Mean reciprocal rank: {0}'.format(np.mean(1./np.array(ranks_left+ranks_right))))

    with open(dir+'/log_file/test.txt', 'wb') as f:
        pickle.dump('Hits tail @{0}: {1}\n'.format(10, np.mean(hits_left[9])), f)
        pickle.dump('Hits head @{0}: {1}\n'.format(10, np.mean(hits_right[9])), f)
        pickle.dump('Hits @{0}: {1}\n'.format(10, np.mean(hits[9])), f)
        pickle.dump('Mean rank tail: {0}\n'.format(np.mean(ranks_left)), f)
        pickle.dump('Mean rank head: {0}\n'.format(np.mean(ranks_right)), f)
        pickle.dump('Mean rank: {0}\n'.format(np.mean(ranks_left+ranks_right)), f)
        pickle.dump('Mean reciprocal rank tail: {0}\n'.format(np.mean(1./np.array(ranks_left))), f)
        pickle.dump('Mean reciprocal rank head: {0}\n'.format(np.mean(1./np.array(ranks_right))), f)
        pickle.dump('Mean reciprocal rank: {0}\n'.format(np.mean(1./np.array(ranks_left+ranks_right))), f)


    f.close()