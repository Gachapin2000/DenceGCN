import os
import argparse
import numpy as np
import statistics
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import f1_score
from torch_geometric.nn import models
from tqdm import tqdm

import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from torch_scatter import scatter
import torch_geometric.transforms as T
from torch_geometric.datasets import PPI
from torch_geometric.data import DataLoader

from models import return_net
from utils import save_conf


def train(epoch, config, loader, model, optimizer, device):
    # train
    model.train()
    criteria = torch.nn.BCEWithLogitsLoss()

    for data in loader: # in [g1, g2, ..., g20]
        data = data.to(device)
        optimizer.zero_grad()
        out, _ = model(data.x, data.edge_index)
        loss = criteria(out, data.y)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def test(config, loader, model, device):
    model.eval()

    ys, preds = [], []
    for data in loader: # only one graph (=g1+g2)
        data = data.to(device)
        ys.append(data.y)
        out, alpha = model(data.x, data.edge_index)
        preds.append((out > 0).float().cpu())

    y    = torch.cat(ys, dim=0).to('cpu').detach().numpy().copy()
    pred = torch.cat(preds, dim=0).to('cpu').detach().numpy().copy()
    return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0


def run(tri, config, data_loader, device):
    train_loader, val_loader, test_loader = data_loader

    model = return_net(config).to(device)
    optimizer = torch.optim.Adam(params       = model.parameters(), 
                                 lr           = config['learning_rate'], 
                                 weight_decay = config['weight_decay'])

    for epoch in tqdm(range(1, config['epochs'])):
        train(epoch, config, train_loader, model, optimizer, device)
    test_acc = test(config, test_loader, model, device)
    
    return test_acc, model


@hydra.main(config_path='conf', config_name='config')
def load(cfg : DictConfig) -> None:
    global config
    config = cfg[cfg.key]

def main():
    global config
    print(config)
    dir_ = './models/{}_{}_{}layer_{}_{}'.format(config['dataset'],config['model'],config['n_layer'],config['jk_mode'],config['att_mode'])
    os.makedirs(dir_)

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

    test_acc = np.zeros(config['n_tri'])
    for tri in range(config['n_tri']):
        test_acc[tri], model = run(tri, config, data_loader, device)
        torch.save(model.state_dict(), dir_ + '/{}th_model.pth'.format(tri))
    
    save_conf(config, dir_ + '/config.txt')
    with open(dir_ + '/acc.txt', 'w') as w:
        w.write('whole test acc ({} tries) = {}'.format(config['n_tri'], test_acc))
        w.write('\tave={:.3f} max={:.3f} min={:.3f}' \
              .format(np.mean(test_acc), np.max(test_acc), np.min(test_acc)))
        
if __name__ == "__main__":
    load()
    main()