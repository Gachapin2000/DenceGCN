import os.path as osp
import argparse
import numpy as np
from tqdm import tqdm
import yaml

import torch
import torch.nn.functional as F
from torch_scatter import scatter
import torch_geometric.transforms as T

from data import FiveUniqueNodes, Planetoid
from models import return_net
from utils import accuracy, HomophilyRank, DictProcessor
from debug import visualize_gat


def train(epoch, config, data, model, optimizer):
    # train
    model.train()
    optimizer.zero_grad()

    # train by class label
    prob_labels = model(data.x, data.edge_index)
    loss_train = F.nll_loss(prob_labels[data.train_mask], data.y[data.train_mask])
    _, correct = accuracy(prob_labels[data.train_mask], data.y[data.train_mask])

    loss_train.backward()
    optimizer.step()

    '''print('Epoch: {:04d}'.format(epoch),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()))'''


def test(config, data, model):
    model.eval()
    prob_labels_test = model(data.x, data.edge_index)
    '''v = visualize_gat(atts, es, data, 18)
    v.visualize()'''
    loss_test = F.nll_loss(prob_labels_test, data.y)

    # top = data.homophily_rank[:5]
    # bot = data.homophily_rank[-5:]
    _, correct = accuracy(prob_labels_test, data.y)
    
    # acc_top = accuracy(prob_labels_test[top], data.y[top])
    # acc_bot = accuracy(prob_labels_test[bot], data.y[bot])

    return correct


def run(config):
    '''print('seed: {}'.format(config.seed))
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False'''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = FiveUniqueNodes(root='../data/toy', 
                              idx_train=[4,13], 
                              x_std=0.25)
    data = dataset[0].to(device)

    config['n_feat']  = data.x.size()[1]
    config['n_class'] = torch.max(data.y).data.item() + 1
    model = return_net(config).to(device)
    optimizer = torch.optim.Adam(params       = model.parameters(), 
                                 lr           = config['learning_rate'], 
                                 weight_decay = config['weight_decay'])

    for epoch in range(1, config['epochs']):
        train(epoch, config, data, model, optimizer)
    correct = test(config, data, model)

    return correct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--key', type=str, default='GATNet_toy')
    parser.add_argument("--override", action=DictProcessor)
    args = parser.parse_args()

    # load parameters of config.yaml
    with open('./config.yaml') as file:
        obj = yaml.safe_load(file)
        config = obj[args.key]

    correct = []
    for tri in range(config['n_tri']):
        correct.append(run(config))
    correct = torch.stack(correct, axis=0)
    whole_correct = torch.mean(correct, axis=0)

    print('config: {}'.format(config))
    for idx, acc in enumerate(whole_correct):
        print('{}'.format(int(acc.data.item()*100)), end=' ')
    print('{:.1f}'.format(int(torch.mean(whole_correct)*100.)))


if __name__ == "__main__":
    main()
