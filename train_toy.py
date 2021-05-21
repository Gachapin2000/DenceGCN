import os.path as osp
import argparse
import numpy as np
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn.functional as F
from torch_scatter import scatter
import torch_geometric.transforms as T

from data import FiveUniqueNodes
from models import return_net
from utils import accuracy, HomophilyRank, DictProcessor
from debug import visualize_gat


def train(epoch, config, data, model, optimizer):
    # train
    model.train()
    optimizer.zero_grad()

    # train by class label
    h, _ = model(data.x, data.edge_index)
    prob_labels = F.log_softmax(h, dim=1)
    loss_train = F.nll_loss(prob_labels[data.train_mask], data.y[data.train_mask])
    _, correct = accuracy(prob_labels[data.train_mask], data.y[data.train_mask])

    loss_train.backward()
    optimizer.step()

    '''print('Epoch: {:04d}'.format(epoch),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()))'''


def test(config, data, model):
    model.eval()
    h, alpha = model(data.x, data.edge_index)
    prob_labels_test = F.log_softmax(h, dim=1)
    '''v = visualize_gat(atts, es, data, 18)
    v.visualize()'''
    loss_test = F.nll_loss(prob_labels_test, data.y)

    # top = data.homophily_rank[:5]
    # bot = data.homophily_rank[-5:]
    _, correct = accuracy(prob_labels_test, data.y)
    
    # acc_top = accuracy(prob_labels_test[top], data.y[top])
    # acc_bot = accuracy(prob_labels_test[bot], data.y[bot])

    return correct, alpha


def run(config):
    '''print('seed: {}'.format(config.seed))
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False'''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = FiveUniqueNodes(root='../data/toy', 
                              split=config['split'], 
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

    return test(config, data, model)


@hydra.main(config_name="./config.yaml")
def load(cfg : DictConfig) -> None:
    global config
    config = cfg[cfg.key]

def main():
    global config
    
    corrects, alphas = [], []
    for tri in range(config['n_tri']):
        correct, alpha = run(config)
        corrects.append(correct)
        alphas.append(alpha)
    correct = torch.stack(corrects, axis=0)
    whole_correct = torch.mean(correct, axis=0)

    print('config: {}'.format(config))
    for idx, acc in enumerate(whole_correct):
        print('{}'.format(int(acc.data.item()*100)), end=' ')
    print('{:.1f}'.format(int(torch.mean(whole_correct)*100.)))

    alpha = alphas[0]
    np.save('./result/{}_JKlstm_{}_layerwise_att_super.npy'.format(config['dataset'], config['att_mode']), alpha.to('cpu').detach().numpy().copy())

if __name__ == "__main__":
    load()
    main()
