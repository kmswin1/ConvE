import torch
import numpy as np
import datetime
from logger import get_logger
from utils import batch_by_size
import time
#timer = CUDATimer()
logger = get_logger('pred', True, True, 'prediction.txt')

# ranking_and_hits(model, Config.batch_size, valid_data, eval_h, eval_t,'dev_evaluation', kg_vocab)
def ranking_and_hits(model, batch_size, dateset, eval_h, eval_t, name, kg_vocab):
    heads, rels, tails = dateset
    logger.info('')
    logger.info('-'*50)
    logger.info(name)
    logger.info('-'*50)
    logger.info('')
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

    for bh, br, bt in batch_by_size(batch_size, heads, rels, tails):
        b_size = bh.size(0)
        bh = bh.cuda();br = br.cuda();bt = bt.cuda()
        pred1 = model.forward(bh, br)
        pred2 = model.forward(bt, br)

        e2_multi1 = torch.empty(b_size, pred1.size(1))
        e2_multi2 = torch.empty(b_size, pred1.size(1))

        for i, (h, r, t) in enumerate(zip(bh, br, bt)):
            e2_multi1[i] = eval_t[h.item(), r.item()].to_dense()
            e2_multi2[i] = eval_h[t.item(), r.item()].to_dense()
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
    idx1 = 0
    idx2 = 0
    for i, meta in enumerate(find_target1):
        if meta == True:
            idx1 = i
    for i, meta in enumerate(find_target2):
        if meta == True:
            idx2 = i
    logger.info ('head : ' + kg_vocab.ent_list[h])
    logger.info("predicted tails : ")
    logger.info(kg_vocab.ent_list[argsort1[idx1]])
    logger.info ('tail : ' + kg_vocab.ent_list[t])
    logger.info(kg_vocab.ent_list[argsort2[idx2]])
    logger.info("predicted heads : ")
    logger.info('Hits tail @{0}: {1}'.format(10, np.mean(hits_left[9])))
    logger.info('Hits head @{0}: {1}'.format(10, np.mean(hits_right[9])))
    logger.info('Hits @{0}: {1}'.format(10, np.mean(hits[9])))
    logger.info('Mean rank tail: {0}'.format(np.mean(ranks_left)))
    logger.info('Mean rank head: {0}'.format(np.mean(ranks_right)))
    logger.info('Mean rank: {0}'.format(np.mean(ranks_left+ranks_right)))
    logger.info('Mean reciprocal rank tail: {0}'.format(np.mean(1./np.array(ranks_left))))
    logger.info('Mean reciprocal rank head: {0}'.format(np.mean(1./np.array(ranks_right))))
    logger.info('Mean reciprocal rank: {0}'.format(np.mean(1./np.array(ranks_left+ranks_right))))
