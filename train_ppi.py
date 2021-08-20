import enum
import os
import argparse
import numpy as np
import statistics
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import f1_score
from torch_geometric.nn import models
from tqdm import tqdm
import mlflow

import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from torch_scatter import scatter
import torch_geometric.transforms as T
from torch_geometric.datasets import PPI
from torch_geometric.data import DataLoader

from models import return_net
from utils import log_params_from_omegaconf_dict


def train(epoch, config, loader, model, optimizer, device):
    # train
    model.train()
    criteria = torch.nn.BCEWithLogitsLoss()

    num_batches = len(loader)
    for batch_id, data in enumerate(loader): # in [g1, g2, ..., g20]
        data = data.to(device)
        optimizer.zero_grad()
        out, _ = model(data.x, data.edge_index)
        loss = criteria(out, data.y)
        loss.backward()
        optimizer.step()
        mlflow.log_metric('loss', value=loss.item(), step=epoch*num_batches + batch_id)
        

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
    
    return test_acc


@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig):
    config = cfg[cfg.key]
    mlflow_runname = cfg.mlflow.runname

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = '~/Study/python/DenceGCN/data/{}_{}'.format(config['dataset'], config['pre_transform'])
    
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    train_dataset = PPI(root.lower(), split='train')
    val_dataset   = PPI(root.lower(), split='val')
    test_dataset  = PPI(root.lower(), split='test')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    data_loader = [train_loader, val_loader, test_loader]

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(mlflow_runname)
    with mlflow.start_run():
        log_params_from_omegaconf_dict(config)
        test_acc = np.zeros(config['n_tri'])
        for tri in range(config['n_tri']):
            test_acc[tri] = run(tri, config, data_loader, device)
            mlflow.log_metric('acc', value=test_acc[tri], step=tri)
        mlflow.log_metric('acc_mean', value=np.mean(test_acc))
        mlflow.log_metric('acc_max', value=np.max(test_acc))
        mlflow.log_metric('acc_min', value=np.min(test_acc))

    return np.mean(test_acc)
    

if __name__ == "__main__":
    main()