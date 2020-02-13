import json
from itertools import count
from collections import namedtuple
import pickle

def load_kg():
    with open('data/ent_str2id', 'rb') as f:
        ent_str2id = pickle.load(f)
    with open('data/ent_id2str', 'rb') as f:
        ent_id2str = pickle.load(f)
    with open('data/rel_str2id', 'rb') as f:
        rel_str2id = pickle.load(f)
    with open('data/rel_id2str', 'rb') as f:
        rel_id2str = pickle.load(f)

    return ent_str2id, ent_id2str, rel_str2id, rel_id2str