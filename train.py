import os.path as osp
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch_scatter import scatter
import torch_geometric.transforms as T

from data import Planetoid
from models import DenceGCN, JKGCN, JKNet, GATNet, GCN
from utils import accuracy, HomophilyRank


def train(epoch, args, data, model, optimizer):
    # train
    model.train()
    optimizer.zero_grad()

    # train by class label
    prob_labels = model(data.x, data.edge_index)
    loss_train = F.nll_loss(prob_labels[data.train_mask], data.y[data.train_mask])
    acc_train  = accuracy(prob_labels[data.train_mask], data.y[data.train_mask])

    loss_train.backward()
    optimizer.step()

    # validation
    model.eval()
    prob_labels_val = model(data.x, data.edge_index)
    loss_val = F.nll_loss(prob_labels_val[data.val_mask], data.y[data.val_mask])
    acc_val = accuracy(prob_labels_val[data.val_mask], data.y[data.val_mask])
    
    print('Epoch: {:04d}'.format(epoch),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()), end=' ')

    return loss_val


def test(args, data, model):
    model.eval()
    prob_labels_test = model(data.x, data.edge_index)
    loss_test = F.nll_loss(prob_labels_test[data.test_mask], data.y[data.test_mask])

    top = data.homophily_rank[:500]
    bot = data.homophily_rank[-500:]
    acc = accuracy(prob_labels_test[data.test_mask], data.y[data.test_mask])
    acc_top = accuracy(prob_labels_test[top], data.y[top])
    acc_bot = accuracy(prob_labels_test[bot], data.y[bot])

    print("Test set results:",
          "loss(test)= {:.4f}".format(loss_test.data.item()),
          "accuracy(test)= {:.4f}".format(acc.data.item()),
          "accuracy(top)= {:.4f}".format(acc_top.data.item()),
          "accuracy(bottom)= {:.4f}".format(acc_bot.data.item()))

    # validate weights for step
    '''step_weight = torch.sum(model.lin.weight, dim=0)
    step_idx = 0
    idx = []
    num_layers = [args.hidden for _ in range(args.layer)]
    for num_layer in num_layers:
        for _ in range(num_layer):
            idx.append(step_idx)
        step_idx += 1
    idx = torch.tensor(idx).to(step_weight.device)
    step_weight = scatter(step_weight, idx, dim=0, reduce='sum')
    print(step_weight)'''

    return acc


def run(args):
    '''print('seed: {}'.format(args.seed))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False'''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Planetoid('../data/{}'.format(args.dataset), args.dataset, split='full', transform=T.NormalizeFeatures())
    data = dataset[0].to(device)
    print(data)

    n_features, n_class = data.x.size()[1], torch.max(data.y).data.item() + 1
    model = GATNet(args.dataset, n_features, args.hidden, args.layer, n_class, args.dropout, 8).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_loss = 100.
    bad_counter = 0
    for epoch in range(1, args.epochs):
        loss_val = train(epoch, args, data, model, optimizer)

        if(loss_val < best_loss):
            best_loss = loss_val
            bad_counter = 0
        else:
            bad_counter += 1
        print('bad_counter: {}'.format(bad_counter))
        if(bad_counter == args.patience):
            break
    
    test_acc = test(args, data, model)

    return test_acc


def main():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--dataset', type=str, default='Cora', help='name of dataset of {Cora, CiteSeer, PubMed}')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
    parser.add_argument('--patience', type=int, default=100, help='the number to stop training.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=12, help='Number of hidden units.')
    parser.add_argument('--layer', type=int, default=8, help='Number of hidden layers.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--aggr', type=str, default='add', help='How to aggregate')
    args = parser.parse_args()

    n_of_tri = 1
    test_acc = np.zeros(n_of_tri)
    for tri in range(n_of_tri):
        test_acc[tri] = run(args)
    print('\twhole test accuracies({} tries) = {}'.format(n_of_tri, test_acc))
    print('\tave: {:.3f} max: {:.3f} min: {:.3f}' \
            .format(np.mean(test_acc), np.max(test_acc), np.min(test_acc)))
    

if __name__ == "__main__":
    main()
