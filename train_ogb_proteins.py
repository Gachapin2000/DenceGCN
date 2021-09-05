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
from torch_geometric.data import RandomNodeSampler
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

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
def test(cfg, loader, model, evaluator, device):
    model.eval()

    ys, preds = [], []
    for data in loader: # only one graph (=g1+g2)
        data = data.to(device)
        out, _ = model(data.x, data.edge_index)
        mask = data['test_mask']
        ys.append(data.y[mask].cpu())
        preds.append(out[mask].cpu())

    test_rocauc = evaluator.eval({
        'y_true': torch.cat(ys, dim=0),
        'y_pred': torch.cat(preds, dim=0),
    })['rocauc']

    return test_rocauc


def run(tri, cfg, data_loader, device):
    train_loader, test_loader = data_loader

    model = return_net(cfg).to(device)
    optimizer = torch.optim.Adam(params       = model.parameters(), 
                                 lr           = cfg['learning_rate'], 
                                 weight_decay = cfg['weight_decay'])
    evaluator = Evaluator('ogbn-proteins')

    for epoch in tqdm(range(1, cfg['epochs'])):
        train(epoch, cfg, train_loader, model, optimizer, device)
    test_acc = test(cfg, test_loader, model, evaluator, device)
    
    return test_acc, model


@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig):
    mlflow_runname = cfg.mlflow.runname
    cfg = cfg[cfg.key]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root='/home/yuru/Study/python/DenceGCN/data/ogbn_proteins'
    
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    dataset = PygNodePropPredDataset('ogbn-proteins', root)
    splitted_idx = dataset.get_idx_split()
    data = dataset[0]
    data.node_species = None
    data.y = data.y.to(torch.float)
    
    # Initialize features of nodes by aggregating edge features.
    row, col = data.edge_index
    data.x = scatter(data.edge_attr, col, 0, dim_size=data.num_nodes, reduce='add')
    cfg.n_feat = cfg.e_feat

    # Set split indices to masks.
    for split in ['train', 'valid', 'test']:
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[splitted_idx[split]] = True
        data[f'{split}_mask'] = mask

    train_loader = RandomNodeSampler(data, num_parts=40, shuffle=True,
                                     num_workers=5)
    test_loader = RandomNodeSampler(data, num_parts=5, num_workers=5)
    data_loader = [train_loader, test_loader]

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