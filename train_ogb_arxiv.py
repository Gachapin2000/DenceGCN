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

    num_batches = len(loader)
    for batch_id, data in enumerate(loader):
        data = data.to(device)
        optimizer.zero_grad()
        out, _ = model(data.x, data.edge_index)
        out = F.log_softmax(out, dim=1)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask].view(-1))
        loss.backward()
        optimizer.step()
        mlflow.log_metric('loss', value=loss.item(), step=epoch*num_batches + batch_id)

        
@torch.no_grad()
def test(cfg, loader, model, evaluator, device):
    model.eval()

    y_true = {'train': [], 'valid': [], 'test': []}
    y_pred = {'train': [], 'valid': [], 'test': []}

    for data in loader: # only one graph (=g1+g2)
        data = data.to(device)
        out, _ = model(data.x, data.edge_index)
        out = out.argmax(dim=-1, keepdim=True)
        for split in y_true.keys():
            mask = data[f'{split}_mask']
            y_true[split].append(data.y[mask].cpu())
            y_pred[split].append(out[mask].cpu())

    train_acc = evaluator.eval({
        'y_true': torch.cat(y_true['train'], dim=0),
        'y_pred': torch.cat(y_pred['train'], dim=0),
    })['acc']

    valid_acc = evaluator.eval({
        'y_true': torch.cat(y_true['valid'], dim=0),
        'y_pred': torch.cat(y_pred['valid'], dim=0),
    })['acc']

    test_acc = evaluator.eval({
        'y_true': torch.cat(y_true['test'], dim=0),
        'y_pred': torch.cat(y_pred['test'], dim=0),
    })['acc']

    return test_acc


def run(tri, cfg, data_loader, device):
    train_loader, test_loader = data_loader

    model = return_net(cfg).to(device)
    optimizer = torch.optim.Adam(params       = model.parameters(), 
                                 lr           = cfg['learning_rate'], 
                                 weight_decay = cfg['weight_decay'])
    evaluator = Evaluator('ogbn-arxiv')

    for epoch in tqdm(range(1, cfg['epochs'])):
        train(epoch, cfg, train_loader, model, optimizer, device)
    test_acc = test(cfg, test_loader, model, evaluator, device)
    
    return test_acc, model


@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig):
    mlflow_runname = cfg.mlflow.runname
    cfg = cfg[cfg.key]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = '~/Study/python/DenceGCN/data/{}_{}'.format(cfg['dataset'], cfg['pre_transform'])
    
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    dataset = PygNodePropPredDataset('ogbn-arxiv', root='../data')
    splitted_idx = dataset.get_idx_split()
    data = dataset[0].to(device)
    
    # Set split indices to masks.
    for split in ['train', 'valid', 'test']:
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[splitted_idx[split]] = True
        data[f'{split}_mask'] = mask
    
    train_loader = RandomNodeSampler(data, num_parts=5, shuffle=True,
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