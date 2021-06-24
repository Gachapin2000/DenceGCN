import os.path as osp
import argparse
import numpy as np
import statistics
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import f1_score
from tqdm import tqdm

import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from torch_scatter import scatter
import torch_geometric.transforms as T
from torch_geometric.datasets import PPI
from torch_geometric.data import DataLoader

from models import return_net
from utils import accuracy, HomophilyRank, HomophilyRank2, summarize_acc


def train(epoch, config, loader, model, optimizer, device):
    # train
    model.train()
    criteria = torch.nn.BCEWithLogitsLoss()

    total_loss = 0.
    alphas = []
    for data in loader: # in [g1, g2, ..., g20]
        data = data.to(device)
        optimizer.zero_grad()
        out, alpha = model(data.x, data.edge_index)
        alphas.append(alpha)
        loss = criteria(out, data.y)
        total_loss += loss.item() * data.num_graphs # num_graphs is always 1
        loss.backward()
        optimizer.step()
    
    return torch.cat(alphas, dim=0)


@torch.no_grad()
def test(config, loader, model, device):
    model.eval()

    alphas = []
    ys, preds = [], []
    for data in loader: # only one graph (=g1+g2)
        data = data.to(device)
        ys.append(data.y)
        out, alpha = model(data.x, data.edge_index)
        alphas.append(alpha)
        preds.append((out > 0).float().cpu())

    y    = torch.cat(ys, dim=0).to('cpu').detach().numpy().copy()
    pred = torch.cat(preds, dim=0).to('cpu').detach().numpy().copy()
    return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0, torch.cat(alphas, dim=0)


def run(data_loader, config, device):
    train_loader, val_loader, test_loader = data_loader

    model = return_net(config).to(device)
    optimizer = torch.optim.Adam(params       = model.parameters(), 
                                 lr           = config['learning_rate'], 
                                 weight_decay = config['weight_decay'])

    for epoch in tqdm(range(1, config['epochs'])):
        alpha_train = train(epoch, config, train_loader, model, optimizer, device)
    acc, alpha_test = test(config, test_loader, model, device)
    whole_acces = summarize_acc(model, [train_loader]+[test_loader]), mode='multi')
    return acc, alpha_train, alpha_test, whole_acces


@hydra.main(config_path='conf', config_name='config')
def load(cfg : DictConfig) -> None:
    global config
    config = cfg[cfg.key]

def main():
    global config
    print('config: {}\n'.format(config))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = './data/{}_{}'.format(config['dataset'], config['pre_transform'])
    
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    train_dataset = PPI(root.lower(), split='train')
    val_dataset   = PPI(root.lower(), split='val')
    test_dataset  = PPI(root.lower(), split='test')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    data_loader = [train_loader, val_loader, test_loader]

    alphas = []
    test_acc = np.zeros(config['n_tri'])
    for tri in range(config['n_tri']):
        acc, alpha_train, alpha_test, whole_acces = run(data_loader, config, device)
        test_acc[tri] = acc
        alphas.append(torch.cat([alpha_train, alpha_test], dim=0))
    print('whole test acc ({} tries) = {}'.format(config['n_tri'], test_acc))
    print('\tave={:.3f} max={:.3f} min={:.3f}' \
          .format(np.mean(test_acc), np.max(test_acc), np.min(test_acc)))

    for tri, alpha in enumerate(alphas):
        np.save('./result/layerwise_att/{}_{}layers_JKlstm_{}_layerwise_att_tri{}.npy'
                .format(config['dataset'], config['n_layer'], config['att_mode'], tri), \
                alpha.to('cpu').detach().numpy().copy())

if __name__ == "__main__":
    load()
    main()