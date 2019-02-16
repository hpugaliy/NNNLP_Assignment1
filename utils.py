from collections import defaultdict
import time
import random
import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader
import copy
from collections import Counter

def read_dataset(filename,w2i,t2i):
    with open(filename, "r") as f:
        for line in f:
            tag, words = line.strip().split(" ||| ")
            yield ([w2i[x] for x in words.lower().split(" ")], t2i[tag])



def load_embeddings(file):
    w2i = defaultdict(lambda: len(w2i))

    vectors = []
    with open(file) as f:
        n, d = f.readline().strip().split()
        w2i['<unk>']
        vectors.append(np.zeros(int(d)))

        for l in f:
            t = l.strip().split()
            w2i[t[0]]
            vectors.append(np.array([float(k) for k in t[1:]]))
    vectors = torch.from_numpy(np.vstack(vectors)).type(torch.FloatTensor)
    return defaultdict(lambda: 0, w2i), vectors

class LengthBasedDataset(Dataset):
    def __init__(self, data,datatype,batch_size=128,min_length=5 ):
        self.data = data
        self.data.sort(key = lambda x : len(x[0]))
        self.batch_size = batch_size
        self.datatype = datatype
        self.min_length = min_length
    def __len__(self):
        if len(self.data)%self.batch_size == 0 :
            return int(len(self.data)/self.batch_size)
        else :
            return (int(len(self.data)/self.batch_size) + 1)
    def __getitem__(self, index):
        datapoints = self.data[index*self.batch_size : min(len(self.data),(index+1)*self.batch_size)]
        labels = [ data[1] for data in datapoints ]
        sentences = [ data[0] for data in datapoints]
        lengths = [len(s) for s in sentences]
        max_len = max(max(lengths),self.min_length )
        sentences = [ s + [0] * (max_len - len(s)) for s in sentences]
        return torch.tensor(sentences).type(self.datatype) , torch.tensor(labels).type(self.datatype),torch.tensor(lengths).type(self.datatype)            

class CharacterDataset(Dataset):
    def __init__(self, data,datatype , max_len = 512):
        self.data = data
        self.datatype = datatype
        self.max_len = max_len
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        sentence,l = self.data[index]
        sentence =  sentence + ([0] * (self.max_len - len(sentence)))
        #print(len(sentence))
        return torch.tensor(sentence).type(self.datatype) , torch.tensor(l).type(self.datatype)
