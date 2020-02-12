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
                self.len += 1
                line = line.strip("\n")
                line = line.split(' ')
                self.head.append(int(line[0]))
                self.rel.append(int(line[1]))
                if (line[0], line[1]) not in self.triple.keys():
                    self.triple[line[0], line[1]] = set()

                self.triple[line[0],line[1]].add(int(line[2]))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        tail = list(self.triple[str(self.head[idx]), str(self.rel[idx])])
        logits = torch.full((self.n_ent, ), self.epsilon)
        logits[tail] = self.smoothed_value

        return self.head[idx], self.rel[idx], logits

class KG_EvalSet(Dataset):
    def __init__(self, file_path, args, n_ent):
        self.n_ent = n_ent
        self.args = args
        self.len = 0
        self.head = []
        self.rel = []
        self.triple = {}
        with open(file_path) as f:
            for line in f:
                #line = json.loads(line)
                self.len += 1
                line = line.strip("\n")
                line = line.split(' ')
                self.head.append(int(line[0]))
                self.rel.append(int(line[1]))
                if (line[0], line[1]) not in self.triple.keys():
                    self.triple[line[0], line[1]] = set()

                self.triple[line[0],line[1]].add(int(line[2]))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        tail = list(self.triple[str(self.head[idx]), str(self.rel[idx])])
        logits = torch.zeros(self.n_ent)
        logits[tail] = 1

        return self.head[idx], self.rel[idx], logits
