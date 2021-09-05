import enum
import os
import argparse
from mlflow.models import model
import numpy as np
import statistics
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import f1_score
from torch_geometric.nn import models
from tqdm import tqdm
import mlflow
from hydra import utils

import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from torch_scatter import scatter
import torch_geometric.transforms as T
from torch_geometric.datasets import PPI
from torch_geometric.data import DataLoader

from models import return_net
from utils import log_params_from_omegaconf_dict


def train(epoch, cfg, loader, model, optimizer, device):
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
def test(cfg, loader, model, device):
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


def run(tri, cfg, data_loader, device):
    train_loader, val_loader, test_loader = data_loader

    model = return_net(cfg).to(device)
    optimizer = torch.optim.Adam(params       = model.parameters(), 
                                 lr           = cfg['learning_rate'], 
                                 weight_decay = cfg['weight_decay'])

    for epoch in tqdm(range(1, cfg['epochs'])):
        train(epoch, cfg, train_loader, model, optimizer, device)
    test_acc = test(cfg, test_loader, model, device)
    
    return test_acc, model


@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig):
    mlflow_runname = cfg.mlflow.runname
    cfg = cfg[cfg.key]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = utils.get_original_cwd() + '/data/' + cfg.dataset
    
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    train_dataset = PPI(root, split='train')
    val_dataset   = PPI(root, split='val')
    test_dataset  = PPI(root, split='test')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    data_loader = [train_loader, val_loader, test_loader]

    mlflow.set_tracking_uri(utils.get_original_cwd() + '/mlruns')
    mlflow.set_experiment(mlflow_runname)
    with mlflow.start_run():
        log_params_from_omegaconf_dict(cfg)
        test_acc = np.zeros(cfg['n_tri'])
        for tri in range(cfg['n_tri']):
            test_acc[tri], model = run(tri, cfg, data_loader, device)
            mlflow.log_metric('acc', value=test_acc[tri], step=tri)
            mlflow.pytorch.log_model(model, artifact_path='{}-th_model'.format(tri))
        mlflow.log_metric('acc_mean', value=np.mean(test_acc))
        mlflow.log_metric('acc_max', value=np.max(test_acc))
        mlflow.log_metric('acc_min', value=np.min(test_acc))

        return np.mean(test_acc)
    
if __name__ == "__main__":
    main()