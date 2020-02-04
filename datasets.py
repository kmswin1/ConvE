from torch.utils.data import Dataset
import torch
import json

class KG_DataSet(Dataset):
    def __init__(self, file_path, kg_vocab, args, n_ent):
        self.kg_vocab = kg_vocab
        self.n_ent = n_ent
        self.args = args
        self.len = 0
        self.head = []
        self.rel = []
        self.tail = []
        self.smoothed_value = 1 - args.label_smoothing
        self.epsilon = 1.0/self.n_ent
        with open(file_path) as f:
            for line in f:
                self.len += 1
                line = json.loads(line)
                self.head.append(self.kg_vocab.ent_id[line['e1']])
                self.rel.append(self.kg_vocab.rel_id[line['rel']])
                temp = []
                for meta in line['e2_e1toe2'].split('@@'):
                    temp.append(kg_vocab.ent_id[meta])
                self.tail.append(temp)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        print (self.tail[idx])
        logits = torch.full((self.n_ent, ), self.epsilon)
        logits[self.tail[idx]] = self.smoothed_value
        return self.head[idx], self.rel[idx], logits

class KG_EvalSet(Dataset):
    def __init__(self, file_path, kg_vocab):
        self.kg_vocab = kg_vocab
        self.len = 0
        self.head = []
        self.rel = []
        self.tail = []
        self.head2 = []
        self.rel_rev = []
        self.tail2 = []
        with open(file_path) as f:
            for line in f:
                self.len += 1
                line = json.loads(line)
                self.head.append(self.kg_vocab.ent_id[line['e1']])
                self.rel.append(self.kg_vocab.rel_id[line['rel']])
                self.tail.append(line['e2_e1toe2'])

                self.head2.append(self.kg_vocab.ent_id[line['e2']])
                self.rel_rev.append(self.kg_vocab.rel_id[line['rel_eval']])
                self.tail2.append(line['e2_e2toe1'])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.head[idx], self.rel[idx], self.tail[idx], self.head2[idx], self.rel_rev[idx], self.tail2[idx]