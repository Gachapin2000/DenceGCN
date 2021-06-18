import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, PPI, Reddit
from torch_geometric.data import DataLoader
from torch_geometric.data import NeighborSampler
from tqdm.std import tqdm
from sklearn.metrics import f1_score
import argparse
import numpy as np

class Net(nn.Module):
    def __init__(self, in_feat, hid_feat, out_feat):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_feat, hid_feat)
        self.fc2 = nn.Linear(hid_feat, out_feat)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def train(loader, use_batch_id, model, optimizer, device):
    # train
    model.train()
    criteria = torch.nn.BCEWithLogitsLoss()

    ys, preds = [], []
    for batch_id, data in enumerate(loader): # in [g1, g2, ..., g20]
        if batch_id == use_batch_id:
            data = data.to(device)
            ys.append(data.y)
            optimizer.zero_grad()
            out = model(data.x)
            loss = criteria(out, data.y)
            loss.backward()
            optimizer.step()
            preds.append((out > 0).float().cpu())
    
    y    = torch.cat(ys, dim=0).to('cpu').detach().numpy().copy()
    pred = torch.cat(preds, dim=0).to('cpu').detach().numpy().copy()
    return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0

@torch.no_grad()
def test(loader, model, device):
    model.eval()

    ys, preds = [], []
    for data in loader: # only one graph (=g1+g2)
        data = data.to(device)
        ys.append(data.y)
        out = model(data.x)
        preds.append((out > 0).float().cpu())

    y    = torch.cat(ys, dim=0).to('cpu').detach().numpy().copy()
    pred = torch.cat(preds, dim=0).to('cpu').detach().numpy().copy()
    return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0



parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of GNN')
parser.add_argument('-d', '--dataset', type=str, default='Cora')
args = parser.parse_args()

dataset_name = args.dataset
root =  '../data/' + dataset_name.lower() + '_none'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:

    acces_train, acces_test = np.zeros(20), np.zeros(20)
    for tri in range(20):
        torch.manual_seed(tri)
        torch.cuda.manual_seed(tri)
        np.random.seed(tri)
        dataset = Planetoid(root          = root,
                            name          = dataset_name,
                            split         = 'public')
        data = dataset[0].to(device)
        n_feature, n_class = data.x.size()[1], torch.max(data.y).item()+1
        n_hid = 512

        net = Net(n_feature, n_hid, n_class).to(device)
        optimizer = optim.SGD(net.parameters(), lr=0.01)

        for i in tqdm(range(3000)):
            optimizer.zero_grad()
            output = net(data.x)
            output = F.log_softmax(output, dim=-1)
            loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
        acc_train = accuracy(net(data.x)[data.train_mask], data.y[data.train_mask])
        net.eval()
        acc_test = accuracy(net(data.x)[data.test_mask], data.y[data.test_mask])

        print('{}-th batch train/test acc: {}/{}'.format(tri, acc_train, acc_test))
        acces_train[tri] = acc_train
        acces_test[tri] = acc_test
    
    print('whole train acc (20 cross val): {}'.format(acces_train))   
    print('\tmean: {}, max: {}, min: {}'.format(np.mean(acces_train), np.max(acces_train), np.min(acces_train)))
    print('whole test acc (20 cross val): {}'.format(acces_test))
    print('\tmean: {}, max: {}, min: {}'.format(np.mean(acces_test), np.max(acces_test), np.min(acces_test)))   
    print('whole difference between train/test: {}'.format([acc_train-acc_test for acc_train, acc_test in zip(acces_train, acces_test)]))


elif dataset_name == 'PPI':
    train_dataset = PPI(root, split='train')
    val_dataset   = PPI(root, split='val')
    test_dataset  = PPI(root, split='test')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    use_batch_ids = range(20)
    acces_train, acces_test = np.zeros(20), np.zeros(20)
    for use_batch_id in use_batch_ids:
        net = Net(50, 64, 121).to(device)
        optimizer = torch.optim.Adam(params       = net.parameters(), 
                                     lr           = 0.005, 
                                     weight_decay = 0.)
        for i in tqdm(range(3000)):
            acc_train = train(train_loader, use_batch_id, net, optimizer, device)
        net.eval()
        acc_test = test(test_loader, net, device)
        print('{}-th batch train/test acc: {}/{}'.format(use_batch_id, acc_train, acc_test))
        acces_train[use_batch_id] = acc_train
        acces_test[use_batch_id] = acc_test 

    print('whole train acc (20 cross val): {}'.format(acces_train))   
    print('\tmean: {}, max: {}, min: {}'.format(np.mean(acces_train), np.max(acces_train), np.min(acces_train)))   
    print('whole test acc (20 cross val): {}'.format(acces_test))
    print('\tmean: {}, max: {}, min: {}'.format(np.mean(acces_test), np.max(acces_test), np.min(acces_test)))   
    print('whole difference between train/test: {}'.format([acc_train-acc_test for acc_train, acc_test in zip(acces_train, acces_test)]))


else: # dataset == 'Reddit'
    dataset = Reddit(root=root)
    data = dataset[0].to(device)
    n_feature, n_class = data.x.size()[1], torch.max(data.y).item()+1
    n_hid = 256

    net = Net(n_feature, n_hid, n_class).to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    for i in tqdm(range(3000)):
        optimizer.zero_grad()
        output = net(data.x)
        output = F.log_softmax(output, dim=-1)
        loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    acc_train = accuracy(net(data.x)[data.train_mask], data.y[data.train_mask])

    net.eval()
    acc_test = accuracy(net(data.x)[data.test_mask], data.y[data.test_mask])

    print('acc_train: {}'.format(acc_train))
    print('acc_test: {}'.format(acc_test))
    