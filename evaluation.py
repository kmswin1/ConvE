import torch
import numpy as np
import datetime, os
from torch.utils.data import DataLoader
dir = os.getcwd()
def ranking_and_hits(model, args, evalloader, n_ent, kg_vocab, epoch):
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

    for i, data in enumerate(evalloader):
        head, rel, tail, head2, rel_rev, tail2 = data
        head = torch.LongTensor(head)
        rel = torch.LongTensor(rel)
        tail = [torch.LongTensor(vec) for vec in tail]
        head2 = torch.LongTensor(head2)
        rel_rev = torch.LongTensor(rel_rev)
        tail2 = [torch.LongTensor(vec) for vec in tail2]
        head = head.cuda()
        rel = rel.cuda()
        head2 = head2.cuda()
        rel_rev = rel_rev.cuda()
        batch_size = head.size(0)

        e2_multi1 = tail.cuda()
        e2_multi2 = tail2.cuda()
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

    # for i in range(10):
    #     logger.info('Hits left @{0}: {1}'.format(i+1, np.mean(hits_left[i])))
    #     logger.info('Hits right @{0}: {1}'.format(i+1, np.mean(hits_right[i])))

    print('Hits tail @{0}: {1}'.format(10, np.mean(hits_left[9])))
    print('Hits head @{0}: {1}'.format(10, np.mean(hits_right[9])))
    print('Hits @{0}: {1}'.format(10, np.mean(hits[9])))
    print('Mean rank tail: {0}'.format(np.mean(ranks_left)))
    print('Mean rank head: {0}'.format(np.mean(ranks_right)))
    print('Mean rank: {0}'.format(np.mean(ranks_left+ranks_right)))
    print('Mean reciprocal rank tail: {0}'.format(np.mean(1./np.array(ranks_left))))
    print('Mean reciprocal rank head: {0}'.format(np.mean(1./np.array(ranks_right))))
    print('Mean reciprocal rank: {0}'.format(np.mean(1./np.array(ranks_left+ranks_right))))

    with open(dir+'/log_file/log.txt', 'a') as f:
        f.write(str(epoch) + '-----evaluation-----\n')
        f.write('Hits tail @{0}: {1}\n'.format(10, np.mean(hits_left[9])))
        f.write('Hits head @{0}: {1}\n'.format(10, np.mean(hits_right[9])))
        f.write('Hits @{0}: {1}\n'.format(10, np.mean(hits[9])))
        f.write('Mean rank tail: {0}\n'.format(np.mean(ranks_left)))
        f.write('Mean rank head: {0}\n'.format(np.mean(ranks_right)))
        f.write('Mean rank: {0}\n'.format(np.mean(ranks_left+ranks_right)))
        f.write('Mean reciprocal rank tail: {0}\n'.format(np.mean(1./np.array(ranks_left))))
        f.write('Mean reciprocal rank head: {0}\n'.format(np.mean(1./np.array(ranks_right))))
        f.write('Mean reciprocal rank: {0}\n'.format(np.mean(1./np.array(ranks_left+ranks_right))))

    f.close()