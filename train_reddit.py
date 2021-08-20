import os
import hydra
from omegaconf import DictConfig
import numpy as np
from tqdm import tqdm
import mlflow

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Reddit
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import SAGEConv

from models import return_net
from utils import log_params_from_omegaconf_dict

def train(epoch, config, data, train_loader, model, optimizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()

    num_batches = len(train_loader)
    for batch_id, (batch_size, n_id, adjs) in enumerate(train_loader):
        adjs = [adj.to(device) for adj in adjs]
        optimizer.zero_grad()
        h, _ = model(data.x[n_id], adjs, batch_size)
        prob_labels = F.log_softmax(h, dim=1)
        loss = F.nll_loss(prob_labels, data.y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()
        mlflow.log_metric('loss', value=loss.item(), step=epoch*num_batches + batch_id)

@torch.no_grad()
def test(config, data, test_loader, model, optimizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    if config['n_layer'] >= 3: # partial aggregate
        total_correct = 0
        for batch_size, n_id, adjs in test_loader:
            adjs = [adj.to(device) for adj in adjs]
            h, alpha = model(data.x[n_id], adjs, batch_size)
            prob_labels = F.log_softmax(h, dim=1)
            total_correct += int(prob_labels.argmax(dim=-1).eq(data.y[n_id[:batch_size]]).sum())
        test_acc = total_correct / int(data.test_mask.sum())

    else: # full aggregate
        h, alpha = model.inference(data.x, test_loader)
        y_true = data.y.unsqueeze(-1)
        y_pred = h.argmax(dim=-1, keepdim=True)
        test_acc = int(y_pred[data.test_mask].eq(y_true[data.test_mask]).sum()) / int(data.test_mask.sum())
    
    return test_acc


def run(tri, config, data, train_loader, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = return_net(config).to(device)
    optimizer = torch.optim.Adam(params       = model.parameters(), 
                                 lr           = config['learning_rate'], 
                                 weight_decay = config['weight_decay'])
    
    for epoch in tqdm(range(1, config['epochs'])):
        train(epoch, config, data, train_loader, model, optimizer)
    test_acc = test(config, data, test_loader, model, optimizer)

    return test_acc


@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig):
    config = cfg[cfg.key]
    mlflow_runname = cfg.mlflow.runname

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = '~/Study/python/DenceGCN/data/{}_{}'.format(config['dataset'], config['pre_transform'])
    dataset = Reddit(root=root.lower())
    data = dataset[0].to(device)

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    sizes_l = [25, 10, 10, 10, 10, 10] # sampling size of each layer when aggregates
    train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask,
                                   sizes=sizes_l[:config['n_layer']], batch_size=1024, shuffle=False,
                                   num_workers=6) 
    if config['n_layer'] >= 3: # partial aggregate due to gpu memory constraints
        test_loader  = NeighborSampler(data.edge_index, node_idx=data.test_mask,
                                       sizes=sizes_l[:config['n_layer']], batch_size=1024, shuffle=False,
                                       num_workers=6)
    else: # if n_layer <=2, full aggregate
        test_loader = NeighborSampler(data.edge_index, node_idx=None,
                                      sizes=[-1], batch_size=1024, shuffle=False,
                                      num_workers=6)

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(mlflow_runname)
    with mlflow.start_run():
        log_params_from_omegaconf_dict(config)
        test_acc = np.zeros(config['n_tri'])
        for tri in range(config['n_tri']):
            test_acc[tri] = run(tri, config, data, train_loader, test_loader)
            mlflow.log_metric('acc', value=test_acc[tri], step=tri)
        mlflow.log_metric('acc_mean', value=np.mean(test_acc))
        mlflow.log_metric('acc_max', value=np.max(test_acc))
        mlflow.log_metric('acc_min', value=np.min(test_acc))

    return np.mean(test_acc)
        
        
if __name__ == "__main__":
    main()
