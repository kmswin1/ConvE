import json
from itertools import count
from collections import namedtuple

def make_kg_vocab(*data):
    ent_str2id = {}
    ent_id2str = {}
    rel_str2id = {}
    rel_id2str = {}
    for filename in data:
        with open(filename) as f:
            for line in f:
                line = json.loads(line)

    return