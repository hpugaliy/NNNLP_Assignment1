from collections import defaultdict
import time
import random
import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader
import copy
from collections import Counter
#import matplotlib.pyplot as plt
from utils import *
import argparse



class CNNMultiFilter(torch.nn.Module):
    def __init__(self, embeddings, num_filters, window_sizes, ntags, finetune=False,dropout=False):
        super(CNNMultiFilter, self).__init__()

        """ layers """
        self.embedding = torch.nn.Embedding.from_pretrained(embeddings, freeze=not finetune)
        self.num_filters = num_filters
        emb_size = embeddings.size()[1]
        # print(emb_size)
        # uniform initialization
        # torch.nn.init.uniform_(self.embedding.weight, -0.25, 0.25)
        # Conv 1d
        self.convs = torch.nn.ModuleList(
            [torch.nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(k, emb_size), \
                             stride=1, padding=0, dilation=1, groups=1, bias=True) for k in window_sizes])

        self.do2dDropout = dropout
        if self.do2dDropout :
            self.dropout2 = torch.nn.ModuleList(
                [torch.nn.Dropout2d(p=0.5) for k in window_sizes])
        self.dropout = torch.nn.Dropout(0.5)
        self.relu = torch.nn.ReLU()
        self.projection_layer = torch.nn.Linear(in_features=num_filters * len(window_sizes), out_features=ntags,
                                                bias=True)
        # Initializing the projection layer
        torch.nn.init.xavier_uniform_(self.projection_layer.weight)

    def forward(self, words):
        emb = self.embedding(words).permute(1, 0, 2, 3)
        # nwords x emb_size
        # print(emb.shape,emb.size())
        # emb = emb.unsqueeze(1)     # 1 x emb_size x nwords
        #

        conv_op = [conv(emb) for conv in self.convs]
        if self.do2dDropout :
            conv_op = [d(c) for c,d in zip(conv_op,self.dropout2)]
        act_op = [torch.squeeze(self.relu(c), dim=3) for c in conv_op]

        # 1 x num_filters x nwords
        # Do max pooling
        pool_op = [h.max(dim=2)[0] for h in act_op]

        proj_input = torch.cat(pool_op, dim=1)
        proj_input = self.dropout(proj_input)
        out = self.projection_layer(proj_input)  # size(out) = 1 x ntags
        return out


def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    model.to(device)

    running_loss = 0.0
    total_predictions = 0.0
    correct_predictions = 0.0

    start_time = time.time()
    for batch_idx, (data, target, lengths) in enumerate(train_loader):
        #         if batch_idx%200 == 0:
        #             print(batch_idx,"- Time :", time.time()-start_time, 's')
        optimizer.zero_grad()
        data = data.to(device)
        target = target.long().to(device).view(-1)

        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total_predictions += target.size(0)
        correct_predictions += (predicted == target).sum().item()
        loss = criterion(outputs, target)
        running_loss += loss.item()

        loss.backward()
        optimizer.step()

    end_time = time.time()

    running_loss /= len(train_loader)
    acc = (correct_predictions / total_predictions) * 100.0
    print('Training Loss: ', running_loss, 'Time: ', end_time - start_time, 's')
    print('Training Accuracy: ', acc, '%')
    return running_loss


def test_model(model, test_loader, criterion, predict=False,print_stats=True):
    with torch.no_grad():
        model.eval()
        model.to(device)

        running_loss = 0.0
        total_predictions = 0.0
        correct_predictions = 0.0
        predictions = None
        len_wrong_counter = Counter()
        class_wrong_counter = Counter()

        for batch_idx, (data, target, lengths) in enumerate(test_loader):

            data = data.to(device)
            lengths = lengths.view(-1)

            outputs = model(data)

            _, predicted = torch.max(outputs.data, 1)
            if not predict:
                target = target.long().to(device).view(-1)
                total_predictions += target.size(0)
                class_wrong_counter.update([t.item() for t in target[predicted != target].cpu()])
                len_wrong_counter.update([s.item() for s in lengths[(predicted != target).cpu()]])
                correct_predictions += (predicted == target).sum().item()
                loss = criterion(outputs, target).detach()
                running_loss += loss.item()
            else:
                if predictions is not None:
                    predictions = torch.cat((predictions, predicted))
                else:
                    predictions = predicted

        if not predict:
            running_loss /= len(test_loader)
            acc = (correct_predictions / total_predictions) * 100.0
            if print_stats:
                print('Testing Loss: ', running_loss)
                print('Testing Accuracy: ', acc, '%')
            return running_loss, acc, class_wrong_counter, len_wrong_counter
        else:
            print(predictions.size())
            return predictions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str,help="Base Folder of the dataset",required=True)
    parser.add_argument("-e", "--embed", type=str, help="Path to fasttext or other embeddings",required=True)
    parser.add_argument("-cf", "--cnn_filters", type=int, default=100,help="Number of filters to be used in CNN")
    parser.add_argument('--finetune', dest='finetune', action='store_true',help="Option to enable finetuning of Fasttext embeddings")
    parser.set_defaults(finetune=False)
    parser.add_argument('--dropout', dest='dropout', action='store_true',help="Option to enable dropout on convolution feature maps")
    parser.set_defaults(dropout=False)
    parser.add_argument("-f", "--filters", type=str, help="Comma seperated list of filter sizes",default="3,4,5")
    parser.add_argument("-ep", "--epochs", type=int, default=5 , help="Number of epochs to run")
    args = parser.parse_args()
    return args


args = parse_args()

file = args.embed
w2i,embeddings =  load_embeddings(file)
t2i = defaultdict(lambda: len(t2i))
base_path = args.dataset
# Read in the data

train = list(read_dataset(base_path+"/topicclass_train.txt",w2i,t2i))
dev = list(read_dataset(base_path+"/topicclass_valid.txt",w2i,t2i))
nwords = len(w2i)
ntags = len(t2i)
test = list(read_dataset(base_path+"/topicclass_test.txt",w2i,t2i))
type = torch.LongTensor
use_cuda = torch.cuda.is_available()
if use_cuda:
    type = torch.cuda.LongTensor
device = torch.device("cuda" if use_cuda else "cpu")

NFILTERS = args.cnn_filters
FILTERS = [int(k) for k in args.filters.strip().split(",")]
FINETUNE = args.finetune
DROPOUT = args.dropout

train_dataset = LengthBasedDataset(train,torch.LongTensor,min_length=max(FILTERS))
dev_dataset = LengthBasedDataset(dev,torch.LongTensor,min_length=max(FILTERS))
test_dataset = LengthBasedDataset(test,torch.LongTensor,min_length=max(FILTERS))
dataloader_args = dict(shuffle=True, batch_size= 1,num_workers=4, pin_memory=True) if use_cuda \
                        else dict(shuffle=True, batch_size=1)
testloader_args = dict(batch_size= 1,num_workers=4, pin_memory=True) if use_cuda \
                        else dict(batch_size=1)
train_loader = DataLoader(train_dataset, **dataloader_args)
dev_loader = DataLoader(dev_dataset, **testloader_args)
test_loader = DataLoader(test_dataset, **testloader_args)



model = CNNMultiFilter(embeddings,NFILTERS,FILTERS,ntags,finetune=FINETUNE,dropout=DROPOUT)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

n_epochs = args.epochs
Train_loss = []
Test_loss = []
Test_acc = []
best_acc = 0.0
best_model = None
best_model_wts = None

for i in range(n_epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer)
    test_loss, test_acc , class_err , len_err = test_model(model, dev_loader, criterion)
    if test_acc > best_acc :
        best_acc = test_acc
        best_model = (class_err,len_err)
        best_model_wts = copy.deepcopy(model.state_dict())
    Train_loss.append(train_loss)
    Test_loss.append(test_loss)
    Test_acc.append(test_acc)
    print('='*20)

model.load_state_dict(best_model_wts)

pred = test_model(model,dev_loader,criterion,predict=True)
arr = pred.cpu().numpy()
i2t = {value:key for (key,value) in t2i.items()}
labels = [i2t[v] for v in arr]
with open('dev_predictions.txt', 'w') as f:
    for l in labels:
        f.write('%s\n' % l)
pred = test_model(model,test_loader,criterion,predict=True)
arr = pred.cpu().numpy()
#i2t = {value:key for (key,value) in t2i.items()}
labels = [i2t[v] for v in arr]
with open('test_predictions.txt', 'w') as f:
    for l in labels:
        f.write('%s\n' % l)
