import os.path as osp
import argparse
import numpy as np
import statistics
import hydra
from omegaconf import DictConfig, OmegaConf

import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from torch_scatter import scatter
import torch_geometric.transforms as T

from data import Planetoid
from models import return_net
from utils import accuracy, HomophilyRank


def train(epoch, config, data, model, optimizer):
    # train
    model.train()
    optimizer.zero_grad()

    # train by class label
    h = model(data.x, data.edge_index)
    prob_labels = F.log_softmax(h, dim=1)
    loss_train = F.nll_loss(
        prob_labels[data.train_mask], data.y[data.train_mask])
    acc_train, _ = accuracy(
        prob_labels[data.train_mask], data.y[data.train_mask])

    loss_train.backward()
    optimizer.step()

    # validation
    model.eval()
    h = model(data.x, data.edge_index)
    prob_labels_val = F.log_softmax(h, dim=1)
    loss_val = F.nll_loss(
        prob_labels_val[data.val_mask], data.y[data.val_mask])
    acc_val, _ = accuracy(
        prob_labels_val[data.val_mask], data.y[data.val_mask])

    '''print('Epoch: {:04d}'.format(epoch),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()), end=' ')'''

    return loss_val


def test(config, data, model):
    model.eval()
    h = model(data.x, data.edge_index)
    prob_labels_test = F.log_softmax(h, dim=1)
    loss_test = F.nll_loss(
        prob_labels_test[data.test_mask], data.y[data.test_mask])

    top = data.homophily_rank['avg'][:500]
    bot = data.homophily_rank['avg'][-500:]
    acc, _ = accuracy(prob_labels_test[data.test_mask], data.y[data.test_mask])
    acc_top, _ = accuracy(prob_labels_test[top], data.y[top])
    acc_bot, _ = accuracy(prob_labels_test[bot], data.y[bot])

    '''print("Test set results:",
          "loss(test)= {:.4f}".format(loss_test.data.item()),
          "accuracy(test)= {:.4f}".format(acc.data.item()),
          "accuracy(top)= {:.4f}".format(acc_top.data.item()),
          "accuracy(bottom)= {:.4f}".format(acc_bot.data.item()))'''

    return acc, acc_top, acc_bot


def run(data, config, device):
    model = return_net(config).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])

    best_loss = 100.
    bad_counter = 0
    for epoch in range(1, config['epochs']):
        loss_val = train(epoch, config, data, model, optimizer)

        if(loss_val < best_loss):
            best_loss = loss_val
            bad_counter = 0
        else:
            bad_counter += 1
        if(bad_counter == config['patience']):
            break

    return test(config, data, model)


@hydra.main(config_name="./config.yaml")
def load(cfg: DictConfig) -> None:
    global config
    config = cfg[cfg.key]


def main():
    global config

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = './data/{}_{}'.format(config['dataset'], config['pre_transform'])
    dataset = Planetoid(root=root.lower(),
                        name=config['dataset'],
                        split=config['split'],
                        transform=eval(config['transform']),
                        pre_transform=eval(config['pre_transform']))
    data = dataset[0].to(device)

    test_acc = np.zeros(config['n_tri'])
    test_acc_top = np.zeros(config['n_tri'])
    test_acc_bot = np.zeros(config['n_tri'])
    # alphas = []
    for tri in range(config['n_tri']):
        test_acc[tri], test_acc_top[tri], test_acc_bot[tri] = run(
            data, config, device)
        # alphas.append(alpha)
    print('config: {}\n'.format(config))
    for acc_criteria in ['test_acc', 'test_acc_top', 'test_acc_bot']:
        acc = eval(acc_criteria)
        print('whole {} ({} tries) = {}'.format(
            acc_criteria, config['n_tri'], acc))
        print('\tave={:.3f} max={:.3f} min={:.3f}'
              .format(np.mean(acc), np.max(acc), np.min(acc)))

    '''best_epoch = np.argmax(test_acc)
    alpha = alphas[best_epoch]
    alpha_avg = alpha[data.homophily_rank['avg']]
    alpha_homo = alpha[data.homophily_rank['homo']]
    alpha_hetero = alpha[data.homophily_rank['hetero']]
    np.save('./result/{}_JKlstm_{}_layerwise_att.npy'.format(config['dataset'], config['att_mode']), alpha.to('cpu').detach().numpy().copy())
    np.save('./result/{}_JKlstm_{}_layerwise_att_avg.npy'.format(config['dataset'], config['att_mode']), alpha_avg.to('cpu').detach().numpy().copy())
    np.save('./result/{}_JKlstm_{}_layerwise_att_homo.npy'.format(config['dataset'], config['att_mode']), alpha_homo.to('cpu').detach().numpy().copy())
    np.save('./result/{}_JKlstm_{}_layerwise_att_hetero.npy'.format(config['dataset'], config['att_mode']), alpha_hetero.to('cpu').detach().numpy().copy())
'''


if __name__ == "__main__":
    load()
    main()
