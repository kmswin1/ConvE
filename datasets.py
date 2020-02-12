from torch.utils.data import Dataset
import torch
import json

class KG_DataSet(Dataset):
    def __init__(self, file_path, args, n_ent):
        self.n_ent = n_ent
        self.args = args
        self.len = 0
        self.head = []
        self.rel = []
        self.smoothed_value = 1 - args.label_smoothing
        self.epsilon = 1.0/self.n_ent
        self.triple = {}
        with open(file_path) as f:
            for line in f:
                #line = json.loads(line)
                line = line.strip("\n")
                line = line.split(' ')
                self.head.append(int(line[0]))
                self.rel.append(int(line[1]))
                if (line[0], line[1]) not in self.triple.keys():
                    self.triple[line[0], line[1]] = set()

                self.triple[line[0],line[1]].add(int(line[2]))


        self.head = torch.LongTensor(self.head)
        self.rel = torch.LongTensor(self.rel)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        tail = list(self.triple[self.head[idx], self.rel[idx]])
        logits = torch.full((self.n_ent, ), self.epsilon)
        logits[tail] = self.smoothed_value

        return self.head[idx], self.rel[idx], logits

class KG_EvalSet(Dataset):
    def __init__(self, file_path, args, n_ent):
        self.len = 0
        self.head = []
        self.rel = []
        self.tail = []
        self.head2 = []
        self.rel_rev = []
        self.tail2 = []
        self.n_ent = n_ent
        with open(file_path) as f:
            for line in f:
                self.len += 1
                line = json.loads(line)
                self.head.append(line['e1'])
                self.rel.append(line['rel'])
                self.tail.append(line['e2_e1toe2'])
                self.head2.append(line['e2'])
                self.rel_rev.append(line['rel_eval'])
                self.tail2.append(line['e1_e2toe1'])
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        logits = torch.zeros(self.n_ent)
        logits[self.tail[idx]] = 1
        logits2 = torch.zeros(self.n_ent)
        logits2[self.tail2[idx]] = 1
        return self.head[idx], self.rel[idx], logits, self.head2[idx], self.rel_rev[idx], logits2
