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


def train(loader, model, optimizer, device):
    # train
    model.train()
    criteria = torch.nn.BCEWithLogitsLoss()

    ys, preds = [], []
    for data in loader: # in [g1, g2, ..., g20]
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



dataset = 'PPI'
root =  '../data/' + dataset.lower() + '_none'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if dataset in ['Cora', 'CiteSeer', 'PubMed']:
    dataset = Planetoid(root          = root,
                        name          = dataset,
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

    print('acc_train: {}'.format(acc_train))
    print('acc_test: {}'.format(acc_test))


elif dataset == 'PPI':
    train_dataset = PPI(root, split='train')
    val_dataset   = PPI(root, split='val')
    test_dataset  = PPI(root, split='test')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    net = Net(50, 64, 121).to(device)
    optimizer = torch.optim.Adam(params       = net.parameters(), 
                                 lr           = 0.005, 
                                 weight_decay = 0.)

    for i in tqdm(range(3000)):
        train(train_loader, net, optimizer, device)
    acc_train = train(train_loader, net, optimizer, device)

    net.eval()
    acc_test = test(test_loader, net, device)

    print('acc train: {}'.format(acc_train))
    print('acc train: {}'.format(acc_test))


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
    