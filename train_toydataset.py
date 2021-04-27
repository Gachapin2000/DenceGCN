import os.path as osp
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch_scatter import scatter
import torch_geometric.transforms as T

from data import FiveUniqueNodes, Planetoid
from models import DenceGCN, JKGCN, JKNet, GATNet, GCN
from utils import accuracy, HomophilyRank


def train(epoch, args, data, model, optimizer):
    # train
    model.train()
    optimizer.zero_grad()

    # train by class label
    prob_labels = model(data.x, data.edge_index)
    loss_train = F.nll_loss(prob_labels[data.train_mask], data.y[data.train_mask])
    correct  = accuracy(prob_labels[data.train_mask], data.y[data.train_mask])

    loss_train.backward()
    optimizer.step()

    '''print('Epoch: {:04d}'.format(epoch),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()))'''


def test(args, data, model):
    model.eval()
    prob_labels_test = model(data.x, data.edge_index)
    loss_test = F.nll_loss(prob_labels_test, data.y)

    # top = data.homophily_rank[:5]
    # bot = data.homophily_rank[-5:]
    correct = accuracy(prob_labels_test, data.y)
    
    # acc_top = accuracy(prob_labels_test[top], data.y[top])
    # acc_bot = accuracy(prob_labels_test[bot], data.y[bot])

    return correct


def run(args):
    '''print('seed: {}'.format(args.seed))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False'''

    dataset = FiveUniqueNodes(root='../data/toy', 
                              idx_train=[4,13], x_std=0.25)
    # dataset = Planetoid('../data/Cora', 'Cora', split='public', transform=T.NormalizeFeatures())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataset[0].to(device)

    n_features, n_class = data.x.size()[1], torch.max(data.y).data.item() + 1
    # model = GCN(n_features, [2,2,2,n_class], dropout=args.dropout).to(device)    
    # model = GATNet('toy', n_features, args.hidden, args.layer, n_class, args.dropout, args.n_heads).to(device)
    '''model = JKNet(num_features = n_features, 
                  num_hiddens  = args.hidden,
                  num_classes  = n_class,
                  num_layers   = args.layer,
                  dropout      = args.dropout,
                  mode         = 'max').to(device)'''
    model = JKGCN(num_features = n_features, 
                  num_hiddens  = 2,
                  num_classes  = n_class,
                  num_layers   = 5,
                  dropout      = args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs):
        train(epoch, args, data, model, optimizer)
    correct = test(args, data, model)

    return correct


def main():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=2, help='Number of hidden units.')
    parser.add_argument('--layer', type=int, default=3, help='Number of hidden layers.')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of multi heads.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--aggr', type=str, default='add', help='How to aggregate')
    args = parser.parse_args()

    n_of_tri = 100
    correct = []
    for tri in tqdm(range(n_of_tri)):
        correct.append(run(args))
    correct = torch.stack(correct, axis=0)
    whole_correct = torch.mean(correct, axis=0)


    for idx, acc in enumerate(whole_correct):
        print('{}'.format(int(acc.data.item()*100)), end=' ')
    print('{:.1f}'.format(int(torch.mean(whole_correct)*100.)))
    

if __name__ == "__main__":
    main()
