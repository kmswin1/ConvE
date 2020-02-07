# !coding=utf8
from __future__ import print_function
from os.path import join
import json
import os
import numpy as np
import sys
import time
from itertools import count
import pickle
import glob
rdm = np.random.RandomState(234234)


def make_knowledge_graph():
    dir = os.getcwd()
    print (dir)
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
    else:
        dataset_name = 'person'
        #dataset_name = 'melon'
    print('Processing dataset')

    base_path = dir+'/data/'
    files = ['train.json', 'test.json']

    data = []
    for file in files:
        with open(join(base_path, file)) as f:
            data = f.readlines() + data


    label_graph = {}
    train_graph = {}
    test_cases = {}
    for file in files:
        test_cases[file] = []
        train_graph[file] = {}


    for file in files:
        with open(join(base_path, file)) as f:
            for i, line in enumerate(f):
                line = json.loads(line)
                e1 = line['src']
                e2 = line['dst']
                rel = line['dstProperty']
                rel_reverse = rel+ '_reverse'

                # data
                # (Mike, fatherOf, John)
                # (John, fatherOf, Tom)

                if (e1 , rel) not in label_graph:
                    label_graph[(e1, rel)] = set()

                if (e2,  rel_reverse) not in label_graph:
                    label_graph[(e2, rel_reverse)] = set()

                if (e1,  rel) not in train_graph[file]:
                    train_graph[file][(e1, rel)] = set()
                if (e2, rel_reverse) not in train_graph[file]:
                    train_graph[file][(e2, rel_reverse)] = set()

                # labels
                # (Mike, fatherOf, John)
                # (John, fatherOf, Tom)
                # (John, fatherOf_reverse, Mike)
                # (Tom, fatherOf_reverse, Mike)
                label_graph[(e1, rel)].add(e2)

                label_graph[(e2, rel_reverse)].add(e1)

                # test cases
                # (Mike, fatherOf, John)
                # (John, fatherOf, Tom)
                test_cases[file].append([e1, rel, e2])

                # data
                # (Mike, fatherOf, John)
                # (John, fatherOf, Tom)
                # (John, fatherOf_reverse, Mike)
                # (Tom, fatherOf_reverse, John)
                train_graph[file][(e1, rel)].add(e2)
                train_graph[file][(e2, rel_reverse)].add(e1)
    return label_graph, train_graph, test_cases


def write_training_graph(cases, graph, path):
    with open(path, 'w') as f:
        n = len(graph)
        for i, key in enumerate(graph):
            e1, rel = key
            # (Mike, fatherOf, John)
            # (John, fatherOf, Tom)
            # (John, fatherOf_reverse, Mike)
            # (Tom, fatherOf_reverse, John)

            # (John, fatherOf) -> Tom
            # (John, fatherOf_reverse, Mike)
            entities1 = "@@".join(list(graph[key]))

            data_point = {}
            data_point['e1'] = e1
            data_point['e2'] = 'None'
            data_point['rel'] = rel
            data_point['rel_eval'] = 'None'
            data_point['e2_e1toe2'] =  entities1
            data_point['e1_e2toe1'] = "None"

            f.write(json.dumps(data_point)  + '\n')

def write_evaluation_graph(cases, graph, path):
    with open(path, 'w') as f:
        n = len(cases)
        n1 = 0
        n2 = 0
        for i, (e1, rel, e2) in enumerate(cases):
            # (Mike, fatherOf) -> John
            # (John, fatherOf, Tom)
            rel_reverse = rel+'_reverse'
            entities1 = "@@".join(list(graph[(e1, rel)]))
            entities2 = "@@".join(list(graph[(e2, rel_reverse)]))

            n1 += len(entities1.split('@@'))
            n2 += len(entities2.split('@@'))


            data_point = {}
            data_point['e1'] = e1
            data_point['e2'] = e2
            data_point['rel'] = rel
            data_point['rel_eval'] = rel_reverse
            data_point['e2_e1toe2'] = entities1
            data_point['e1_e2toe1'] = entities2

            f.write(json.dumps(data_point)  + '\n')

def write_training_graph2idx(cases, graph, ent_str2idx, rel_str2idx, path):
    with open(path, 'w') as f:
        n = len(cases)
        n1 = 0
        n2 = 0
        for i, (e1, rel, e2) in enumerate(cases):
            # (Mike, fatherOf) -> John
            # (John, fatherOf, Tom)
            rel_reverse = rel+'_reverse'
            entities1 = "@@".join(list(graph[(e1, rel)]))
            entities2 = "@@".join(list(graph[(e2, rel_reverse)]))

            n1 += len(entities1.split('@@'))
            n2 += len(entities2.split('@@'))

            ents1 = []
            for meta in entities1.split('@@'):
                ents1.append(ent_str2idx[meta])


            data_point = {}
            data_point['e1'] = ent_str2idx[e1]
            data_point['e2'] = 'None'
            data_point['rel'] = rel_str2idx[rel]
            data_point['rel_eval'] = 'None'
            data_point['e2_e1toe2'] =  ents1
            data_point['e1_e2toe1'] = "None"

            f.write(json.dumps(data_point)  + '\n')

def write_evaluation_graph2idx(cases, graph, ent_str2idx, rel_str2idx, path):
    with open(path, 'w') as f:
        n = len(cases)
        n1 = 0
        n2 = 0
        for i, (e1, rel, e2) in enumerate(cases):
            # (Mike, fatherOf) -> John
            # (John, fatherOf, Tom)
            rel_reverse = rel+'_reverse'
            entities1 = "@@".join(list(graph[(e1, rel)]))
            entities2 = "@@".join(list(graph[(e2, rel_reverse)]))

            n1 += len(entities1.split('@@'))
            n2 += len(entities2.split('@@'))

            ents1 = []
            ents2 = []
            for meta in entities1.split('@@'):
                ents1.append(ent_str2idx[meta])

            for meta in entities2.split('@@'):
                ents2.append(ent_str2idx[meta])

            data_point = {}
            data_point['e1'] = ent_str2idx[e1]
            data_point['e2'] = ent_str2idx[e2]
            data_point['rel'] = rel_str2idx[rel]
            data_point['rel_eval'] = rel_str2idx[rel_reverse]
            data_point['e2_e1toe2'] = ents1
            data_point['e1_e2toe1'] = ents2

            f.write(json.dumps(data_point)  + '\n')

def make_kg_vocab(*data):
    ent_set = set()
    rel_set = set()
    for filename in data:
        with open(filename) as f:
            for line in f:
                line = json.loads(line)
                e1 = line['src']
                e2 = line['dst']
                rel = line['dstProperty']
                rel_rev = line['dstProperty'] + '_reverse'
                ent_set.add(e1)
                ent_set.add(e2)
                rel_set.add(rel)
                rel_set.add(rel_rev)
    ent_list = sorted(list(ent_set))
    rel_list = sorted(list(rel_set))
    ent_dict = dict(zip(ent_list, count()))
    rel_dict = dict(zip(rel_list, count()))

    return ent_list, rel_list, ent_dict, rel_dict

def write_str2id(kg, path):
    with open(path, 'wb') as f:
        str2id = {}
        for i, meta in enumerate(kg):
            str2id[meta] = i

        pickle.dump(str2id, f)

def write_id2str(kg, path):
    with open(path, 'wb') as f:
        id2str = {}
        for i, meta in enumerate(kg):
            id2str[i] = meta

        pickle.dump(id2str, f)

def write_train_set(data, ent_str2id, rel_str2id):
    with open('data/train_set.txt', 'w') as ff:
        with open(data, 'r') as f:
            for line in f:
                line = json.loads(line)
                e1 = line['src']
                e2 = line['dst']
                rel = line['dstProperty']
                rel_rev = line['dstProperty'] + '_reverse'

                ff.write(str(ent_str2id[e1]) + " " + str(rel_str2id[rel]) + " " + str(ent_str2id[e2]) + "\n")
                ff.write(str(ent_str2id[e2]) + " " + str(rel_str2id[rel_rev]) + " " + str(ent_str2id[e1]) + "\n")


def main():
    '''
    label_graph, train_graph, test_cases = make_knowledge_graph()
    start = time.time()
    all_cases = test_cases['train.json'] + test_cases['test.json']
    write_training_graph(test_cases['train.json'], train_graph['train.json'], 'data/e1rel_to_e2_train.json')
    write_evaluation_graph(test_cases['test.json'], label_graph, 'data/e1rel_to_e2_ranking_test.json')
    write_training_graph(all_cases, label_graph, 'data/e1rel_to_e2_full.json')
    '''

    start = time.time()
    label_graph, train_graph, test_cases = make_knowledge_graph()
    ent_list, rel_list, ent_dict, rel_dict = make_kg_vocab('data/train.json', 'data/test.json')
    write_str2id(ent_list, 'data/ent_str2id')
    write_id2str(ent_list, 'data/ent_id2str')
    write_str2id(rel_list, 'data/rel_str2id')
    write_id2str(rel_list, 'data/rel_id2str')
    write_evaluation_graph2idx(test_cases['test.json'], label_graph, ent_dict, rel_dict, 'data/test_ranking.json')
    write_eavaluation_graph2idx('data/train.json', ent_dict, rel_dict)
    print (time.time() - start)

if __name__ == '__main__':
    main()