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


def train(epoch, cfg, data, model, optimizer, device):
    model.train()

    optimizer.zero_grad()
    out, _ = model(data.x, data.adj_t)
    out = out.log_softmax(dim=-1)
    out = out[data['train_mask']]
    loss = F.nll_loss(out, data.y.squeeze(1)[data['train_mask']])
    loss.backward()
    optimizer.step()
    mlflow.log_metric('loss', value=loss.item(), step=epoch)


@torch.no_grad()
def test(cfg, data, model, evaluator, device):
    model.eval()

    out, _ = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[data['train_mask']],
        'y_pred': y_pred[data['train_mask']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[data['valid_mask']],
        'y_pred': y_pred[data['valid_mask']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[data['test_mask']],
        'y_pred': y_pred[data['test_mask']],
    })['acc']

    return test_acc


def run(tri, cfg, data, device):
    model = return_net(cfg).to(device)
    optimizer = torch.optim.Adam(params       = model.parameters(), 
                                 lr           = cfg['learning_rate'], 
                                 weight_decay = cfg['weight_decay'])
    evaluator = Evaluator('ogbn-arxiv')

    for epoch in tqdm(range(1, cfg['epochs'])):
        train(epoch, cfg, data, model, optimizer, device)
    test_acc = test(cfg, data, model, evaluator, device)
    
    return test_acc, model


@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig):
    mlflow_runname = cfg.mlflow.runname
    cfg = cfg[cfg.key]
    print(cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root='/home/yuru/Study/python/DenceGCN/data/ogbn_arxiv'
    
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    dataset = PygNodePropPredDataset('ogbn-arxiv', root, transform=T.ToSparseTensor())
    splitted_idx = dataset.get_idx_split()
    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data = data.to(device)

    # Set split indices to masks.
    for split in ['train', 'valid', 'test']:
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[splitted_idx[split]] = True
        data[f'{split}_mask'] = mask

    mlflow.set_tracking_uri(utils.get_original_cwd() + '/mlruns')
    mlflow.set_experiment(mlflow_runname)
    with mlflow.start_run():
        log_params_from_omegaconf_dict(cfg)
        test_acc = np.zeros(cfg['n_tri'])
        for tri in range(cfg['n_tri']):
            test_acc[tri], model = run(tri, cfg, data, device)
            mlflow.log_metric('acc', value=test_acc[tri], step=tri)
            mlflow.pytorch.log_model(model, artifact_path='{}-th_model'.format(tri))
        mlflow.log_metric('acc_mean', value=np.mean(test_acc))
        mlflow.log_metric('acc_max', value=np.max(test_acc))
        mlflow.log_metric('acc_min', value=np.min(test_acc))

    return np.mean(test_acc)
    

if __name__ == "__main__":
    main()