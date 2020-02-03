# !coding=utf8
from __future__ import print_function
from os.path import join
import json
import os
import numpy as np
import sys
import time
from utils import make_kg_id

rdm = np.random.RandomState(234234)


def make_knowledge_graph(kg_vocab):
    dir = os.getcwd()
    print (dir)
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
                e1 = kg_vocab.ent_id[line['src']]
                e2 = kg_vocab.ent_id[line['dst']]
                rel = kg_vocab.rel_id[line['dstProperty']]
                rel_reverse = kg_vocab.ent_id[rel+ '_reverse']

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


def write_training_graph(cases, graph, path, kg_vocab):
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
            data_point['e2_e2toe1'] = "None"

            f.write(json.dumps(data_point)  + '\n')

def write_evaluation_graph(cases, graph, path, kg_vocab):
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
            data_point['e2_e2toe1'] = entities2

            f.write(json.dumps(data_point)  + '\n')

def main():
    dir = os.getcwd()
    kg_vocab = make_kg_id(dir+'/data/train.json', dir+'/data/test.json')
    label_graph, train_graph, test_cases = make_knowledge_graph(kg_vocab)
    start = time.time()
    all_cases = test_cases['train.json'] + test_cases['test.json']
    write_training_graph(test_cases['train.json'], train_graph['train.json'], 'data/e1rel_to_e2_train.json', kg_vocab)
    write_evaluation_graph(test_cases['test.json'], label_graph, 'data/e1rel_to_e2_ranking_test.json', kg_vocab)
    write_training_graph(all_cases, label_graph, 'data/e1rel_to_e2_full.json')
    print (time.time() - start)

if __name__ == '__main__':
    main()