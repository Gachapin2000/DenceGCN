import os
import argparse
from typing import Dict
import numpy as np
import statistics
import yaml
import hydra
from hydra import utils
from tqdm import tqdm
from omegaconf import DictConfig
import mlflow
from mlflow.tracking import MlflowClient

import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from torch_scatter import scatter
import torch_geometric.transforms as T

from data import Planetoid
from models import return_net
from utils import accuracy, log_params_from_omegaconf_dict

def train(epoch, cfg, data, model, optimizer):
    # train
    model.train()
    optimizer.zero_grad()

    # train by class label
    h, _ = model(data.x, data.edge_index)
    prob_labels = F.log_softmax(h, dim=1)
    loss_train = F.nll_loss(prob_labels[data.train_mask], data.y[data.train_mask])
    loss_train.backward()
    optimizer.step()

    # validation
    model.eval()
    h, _ = model(data.x, data.edge_index)
    prob_labels_val = F.log_softmax(h, dim=1)
    loss_val = F.nll_loss(prob_labels_val[data.val_mask], data.y[data.val_mask])

    mlflow.log_metric('loss', value=loss_val.item(), step=epoch)

    return loss_val

def test(cfg, data, model):
    model.eval()
    h, _ = model(data.x, data.edge_index)
    prob_labels_test = F.log_softmax(h, dim=1)
    acc, _ = accuracy(prob_labels_test[data.test_mask], data.y[data.test_mask])

    return acc


def run(tri, cfg, data, run_info, seed=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = return_net(cfg).to(device)
    optimizer = torch.optim.Adam(params       = model.parameters(), 
                                 lr           = cfg['learning_rate'], 
                                 weight_decay = cfg['weight_decay'])

    best_loss = 100.
    bad_counter = 0
    for epoch in range(1, cfg['epochs']):
        loss_val = train(epoch, cfg, data, model, optimizer)

        if(loss_val < best_loss):
            best_loss = loss_val
            bad_counter = 0
        else:
            bad_counter += 1
        if(bad_counter == cfg['patience']):
            break

    test_acc = test(cfg, data, model)

    return test_acc, model


@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig):
    mlflow_runname = cfg.mlflow.runname
    cfg = cfg[cfg.key]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = utils.get_original_cwd() + '/data/' + cfg.dataset
    print(root)
    dataset = Planetoid(root          = root,
                        name          = cfg['dataset'],
                        seed          = 0,
                        split         = cfg['split'],
                        transform     = eval(cfg['transform']),
                        pre_transform = eval(cfg['pre_transform']))
    data = dataset[0].to(device)

    mlflow.set_tracking_uri(utils.get_original_cwd() + '/mlruns')
    mlflow.set_experiment(mlflow_runname)
    with mlflow.start_run():
        run_info = mlflow.active_run().info
        test_acc = np.zeros(cfg['n_tri'])
        for tri in tqdm(range(cfg['n_tri'])):
            test_acc[tri], model = run(tri, cfg, data, run_info, seed=tri)
            mlflow.log_metric('acc', value=test_acc[tri], step=tri)
            mlflow.pytorch.log_model(model, artifact_path='{}-th_model'.format(tri))
        mlflow.log_metric('acc_mean', value=np.mean(test_acc))
        mlflow.log_metric('acc_max', value=np.max(test_acc))
        mlflow.log_metric('acc_min', value=np.min(test_acc))
        log_params_from_omegaconf_dict(cfg)
        
        return np.mean(test_acc)

if __name__ == "__main__":
    main()