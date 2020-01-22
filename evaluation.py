import torch
import numpy as np
import datetime, os
from logger import get_logger
from utils import batch_by_size
import time
#timer = CUDATimer()
dir = os.getcwd()
# ranking_and_hits(model, Config.batch_size, valid_data, eval_h, eval_t,'dev_evaluation')
def ranking_and_hits(model, batch_size, dateset, dataset_rev, eval_h, eval_t, name):
    heads_rev, rels_rev, tails_rev = dataset_rev
    heads, rels, tails = dateset
    #logger.info('')
    #logger.info('-'*50)
    #logger.info(name)
    #logger.info('-'*50)
    #logger.info('')
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

    for bh, br, brr, bt in batch_by_size(batch_size, heads, rels, rels_rev, tails):
        b_size = bh.size(0)
        bh = bh.cuda();br = br.cuda();bt = bt.cuda();brr = brr.cuda()
        pred1 = model.forward(bh, br)
        pred2 = model.forward(bt, brr)

        e2_multi1 = torch.empty(b_size, pred1.size(1))
        e2_multi2 = torch.empty(b_size, pred1.size(1))

        for i, (h, r, rr, t) in enumerate(zip(bh, br, brr, bt)):
            e2_multi1[i] = eval_t[h.item(), r.item()].to_dense()
            e2_multi2[i] = eval_h[t.item(), rr.item()].to_dense()
        e2_multi1 = e2_multi1.cuda()
        e2_multi2 = e2_multi2.cuda()

        for i in range(b_size):
            # save the prediction that is relevant
            target_value1 = pred1[i,bt[i].item()].item()
            target_value2 = pred2[i,bh[i].item()].item()
            # zero all known cases (this are not interesting)
            # this corresponds to the filtered setting
            pred1[i] += e2_multi1[i] * (-1e20)
            pred2[i] += e2_multi2[i] * (-1e20)
            # write base the saved values
            pred1[i][bt[i].item()] = target_value1
            pred2[i][bh[i].item()] = target_value2

        # sort and rank
        max_values, argsort1 = torch.sort(pred1, 1, descending=True)
        max_values, argsort2 = torch.sort(pred2, 1, descending=True)
        # max_values, argsort2 = torch.sort(pred2, 1, descending=True)
        for i in range(b_size):
            # find the rank of the target entities
            find_target1 = argsort1[i] == bt[i]
            find_target2 = argsort2[i] == bh[i]
            rank1 = torch.nonzero(find_target1)[0, 0].item() + 1
            rank2 = torch.nonzero(find_target2)[0, 0].item() + 1
            # rank+1, since the lowest rank is rank 1 not rank 0
            # ranks.append(rank1+1)
            ranks_left.append(rank1)
            # ranks.append(rank2+1)
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
        f.write('-----evaluation-----\n')
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
