import json
from itertools import count
from collections import namedtuple

def make_kg_vocab(*data):
    kg_vocab = namedtuple('kg_vocab', ['ent_list', 'rel_list', 'ent_id', 'rel_id'])
    ent_set = set()
    rel_set = set()
    for filename in data:
        with open(filename) as f:
            for line in f:
                line = json.loads(line)
                e1 = line['e1']
                rel = line['rel']
                ent_set.add(e1)
                rel_set.add(rel)
    ent_list = sorted(list(ent_set))
    rel_list = sorted(list(rel_set))
    ent_id = dict(zip(ent_list, count()))
    rel_id = dict(zip(rel_list, count()))
    return kg_vocab(ent_list, rel_list, ent_id, rel_id)

def graph_size(vocab):
    return len(vocab.ent_id), len(vocab.rel_id)