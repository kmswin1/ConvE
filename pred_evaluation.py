import torch
import numpy as np
import datetime, os
import pickle, json
dir = os.getcwd()
def ranking_and_hits(model, args, evalloader, n_ent, ent_id2str, rel_id2str):
    hits_left = []
    hits_right = []
    hits = []
    ranks = []
    ranks_left = []
    ranks_right = []
    hit1_cnt = 0
    nohit1_cnt = 0
    hit10_cnt = 0
    nohit10_cnt = 0
    hit1_rels = {}
    nohit1_rels = {}
    hit10_rels = {}
    nohit10_rels = {}

    for i in range(10):
        hits_left.append([])
        hits_right.append([])
        hits.append([])

    for i, data in enumerate(evalloader):
        #head, rel, tail, head2, rel_rev, tail2 = data
        head, rel, tail, head2 = data
        head = torch.LongTensor(head)
        rel = torch.LongTensor(rel)
        #head2 = torch.LongTensor(head2)
        #rel_rev = torch.LongTensor(rel_rev)
        head = head.cuda()
        rel = rel.cuda()
        head2 = head2.cuda()
        #rel_rev = rel_rev.cuda()
        batch_size = head.size(0)

        e2_multi1 = tail.cuda()
        #e2_multi2 = tail2.cuda()
        pred1 = model.forward(head, rel)
        #pred2 = model.forward(head2, rel_rev)

        for i in range(batch_size):
            # save the prediction that is relevant
            target_value1 = pred1[i][head2[i]].item()
            #target_value2 = pred2[i][head[i]].item()
            # zero all known cases (this are not interesting)
            # this corresponds to the filtered setting
            pred1[i][e2_multi1[i] == 1] = 0.0
            #pred2[i][e2_multi2[i] == 1] = 0.0
            # write base the saved values
            pred1[i][head2[i]] = target_value1
            #pred2[i][head[i]] = target_value2

        # sort and rank
        max_values, argsort1 = torch.sort(pred1, 1, descending=True)
        #max_values, argsort2 = torch.sort(pred2, 1, descending=True)
        for i in range(batch_size):
            # find the rank of the target entities
            find_target1 = argsort1[i] == head2[i]
            #find_target2 = argsort2[i] == head[i]
            rank1 = torch.nonzero(find_target1)[0, 0].item() + 1
            #rank2 = torch.nonzero(find_target2)[0, 0].item() + 1
            # rank+1, since the lowest rank is rank 1 not rank 0
            ranks.append(rank1+1)
            ranks_left.append(rank1)
            #ranks.append(rank2+1)
            #ranks_right.append(rank2)

            # this could be done more elegantly, but here you go
            hits[9].append(int(rank1<=10))
            #hits[9].append(int(rank2<=10))
            hits_left[9].append((int(rank1<=10)))
            #hits_right[9].append((int(rank2<=10)))
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
                with open(dir + '/log_file/hit1.txt', 'a') as f:
                    f.write(ent_id2str[head[i].item()]+"\n")
                    f.write(rel_id2str[rel[i].item()]+"\n")
                    f.write(ent_id2str[argsort1[i][0].item()]+"\n")
                    if rel_id2str[rel[i].item()] not in hit1_rels.keys():
                        hit1_rels[rel_id2str[rel[i].item()]] = 1
                    else:
                        hit1_rels[rel_id2str[rel[i].item()]] += 1
                    hit1_cnt += 1
                    data_point = {}
                    top = []
                    for j in range(10):
                        top.append(ent_id2str[argsort1[i][j].item()])
                    data_point['head'] = ent_id2str[head[i].item()]
                    data_point['relation'] = rel_id2str[rel[i].item()]
                    data_point['topten'] = top
                    with open(dir + '/log_file/hit1.json', 'a') as ff:
                        ff.write(json.dumps(data_point) + '\n')

            elif rank1 > 1:
                with open(dir + '/log_file/nohit1.txt', 'a') as f:
                    f.write(ent_id2str[head[i].item()]+"\n")
                    f.write(rel_id2str[rel[i].item()]+"\n")
                    f.write(ent_id2str[argsort1[i][0].item()]+"\n")
                    if rel_id2str[rel[i].item()] not in nohit1_rels.keys():
                        nohit1_rels[rel_id2str[rel[i].item()]] = 1
                    else:
                        nohit1_rels[rel_id2str[rel[i].item()]] += 1
                    nohit1_cnt += 1
                    data_point = {}
                    top = []
                    for j in range(10):
                        top.append(ent_id2str[argsort1[i][j].item()])
                    data_point['head'] = ent_id2str[head[i].item()]
                    data_point['relation'] = rel_id2str[rel[i].item()]
                    data_point['topten'] = top
                    with open(dir + '/log_file/nohit1.json', 'a') as ff:
                        ff.write(json.dumps(data_point) + '\n')

            if rank1 <= 10:
                with open(dir + '/log_file/hit10.txt', 'a') as f:
                    f.write(ent_id2str[head[i].item()]+"\n")
                    f.write(rel_id2str[rel[i].item()]+"\n")
                    f.write(ent_id2str[argsort1[i][0].item()]+"\n")
                    if rel_id2str[rel[i].item()] not in hit10_rels.keys():
                        hit10_rels[rel_id2str[rel[i].item()]] = 1
                    else:
                        hit10_rels[rel_id2str[rel[i].item()]] += 1
                    hit10_cnt += 1
                    data_point = {}
                    top = []
                    for j in range(10):
                        top.append(ent_id2str[argsort1[i][j].item()])
                    data_point['head'] = ent_id2str[head[i].item()]
                    data_point['relation'] = rel_id2str[rel[i].item()]
                    data_point['topten'] = top
                    with open(dir + '/log_file/hit10.json', 'a') as ff:
                        ff.write(json.dumps(data_point) + '\n')

            elif rank1 > 10:
                with open(dir + '/log_file/nohit10.txt', 'a') as f:
                    f.write(ent_id2str[head[i].item()]+"\n")
                    f.write(rel_id2str[rel[i].item()]+"\n")
                    f.write(ent_id2str[argsort1[i][0].item()]+"\n")
                    if rel_id2str[rel[i].item()] not in nohit10_rels.keys():
                        nohit10_rels[rel_id2str[rel[i].item()]] = 1
                    else:
                        nohit10_rels[rel_id2str[rel[i].item()]] += 1
                    nohit10_cnt += 1
                    data_point = {}
                    top = []
                    for j in range(10):
                        top.append(ent_id2str[argsort1[i][j].item()])
                    data_point['head'] = ent_id2str[head[i].item()]
                    data_point['relation'] = rel_id2str[rel[i].item()]
                    data_point['topten'] = top
                    with open(dir + '/log_file/nohit10.json', 'a') as ff:
                        ff.write(json.dumps(data_point) + '\n')

    with open(dir + '/log_file/result.txt', 'w') as f:
        f.write("hit1 : " + str(hit1_cnt) + "\n")
        f.write("nohit1 : " + str(nohit1_cnt) + "\n")
        f.write("hit10 : " + str(hit10_cnt) + "\n")
        f.write("nohit10 : " + str(nohit10_cnt) + "\n")
        f.write('Hits tail @{0}: {1}\n'.format(10, np.mean(hits_left[9])))
        #f.write('Hits head @{0}: {1}\n'.format(10, np.mean(hits_right[9])))
        f.write('Hits @{0}: {1}\n'.format(10, np.mean(hits[9])))
        f.write('Mean rank tail: {0}\n'.format(np.mean(ranks_left)))
        #f.write('Mean rank head: {0}\n'.format(np.mean(ranks_right)))
        f.write('Mean rank: {0}\n'.format(np.mean(ranks_left)))
        f.write('Mean reciprocal rank tail: {0}\n'.format(np.mean(1./np.array(ranks_left))))
        #f.write('Mean reciprocal rank head: {0}\n'.format(np.mean(1./np.array(ranks_right))))
        f.write('Mean reciprocal rank: {0}\n'.format(np.mean(1./np.array(ranks_left))))
        
    with open(dir + '/log_file/hit1_rels', 'wb') as f:
        pickle.dump(hit1_rels, f)
    with open(dir + '/log_file/nohit1_rels', 'wb') as f:
        pickle.dump(nohit1_rels, f)
    with open(dir + '/log_file/hit10_rels', 'wb') as f:
        pickle.dump(hit10_rels, f)
    with open(dir + '/log_file/nohit10_rels', 'wb') as f:
        pickle.dump(nohit10_rels, f)

    print('Hits tail @{0}: {1}'.format(10, np.mean(hits_left[9])))
    #print('Hits head @{0}: {1}'.format(10, np.mean(hits_right[9])))
    print('Hits @{0}: {1}'.format(10, np.mean(hits[9])))
    print('Mean rank tail: {0}'.format(np.mean(ranks_left)))
    #print('Mean rank head: {0}'.format(np.mean(ranks_right)))
    print('Mean rank: {0}'.format(np.mean(ranks_left+ranks_right)))
    print('Mean reciprocal rank tail: {0}'.format(np.mean(1./np.array(ranks_left))))
    #print('Mean reciprocal rank head: {0}'.format(np.mean(1./np.array(ranks_right))))
    print('Mean reciprocal rank: {0}'.format(np.mean(1./np.array(ranks_right))))