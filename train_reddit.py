import os.path as osp
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np

import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.datasets import Reddit
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import SAGEConv

from models import return_net


def train(epoch, config, data, train_loader, model, optimizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()

    total_loss = total_correct = 0
    for batch_size, n_id, adjs in train_loader:
        # batch_size is 1024, 
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs] # 2 adj, because of 2 layer-conv

        optimizer.zero_grad()
        # n_id is (107741)(=[v_53030, v_182890, ...]) , it is Batch_0
        h, _ = model(data.x[n_id], adjs, batch_size) # out is (1024, 41)
        prob_labels = F.log_softmax(h, dim=1)
        loss = F.nll_loss(prob_labels, data.y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(prob_labels.argmax(dim=-1).eq(data.y[n_id[:batch_size]]).sum())

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / int(data.train_mask.sum())



@torch.no_grad()
def test(config, data, all_subgraph_loader, model, optimizer):
    model.eval()

    h, alpha = model.inference(data.x, all_subgraph_loader)
    y_true = data.y.unsqueeze(-1)
    y_pred = h.argmax(dim=-1, keepdim=True)

    test_acc = int(y_pred[data.test_mask].eq(y_true[data.test_mask]).sum()) / int(data.test_mask.sum())

    return test_acc, alpha

def run(tri, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = './data/{}_{}'.format(config['dataset'], config['pre_transform'])
    dataset = Reddit(root=root.lower())
    data = dataset[0].to(device)

    train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask,
                                   sizes=[100, 50, 25, 10], batch_size=1024, shuffle=True,
                                   num_workers=12) # sizes is sampling size when aggregates
    all_subgraph_loader = NeighborSampler(data.edge_index, node_idx=None,
                                          sizes=[-1], batch_size=1024, shuffle=False,
                                          num_workers=12) # all nodes is considered

    model = return_net(config).to(device)
    optimizer = torch.optim.Adam(params       = model.parameters(), 
                                 lr           = config['learning_rate'], 
                                 weight_decay = config['weight_decay'])

    
    for epoch in range(1, config['epochs']):
        train(epoch, config, data, train_loader, model, optimizer)
    
    return test(config, data, all_subgraph_loader, model, optimizer)


@hydra.main(config_name="./config.yaml")
def load(cfg : DictConfig) -> None:
    global config
    config = cfg[cfg.key]

def main():
    global config

    test_acces = np.zeros(config['n_tri'])
    alphas = []
    for tri in range(config['n_tri']):
        test_acc, alpha = run(tri, config)
        test_acces[tri] = test_acc
        alphas.append(alpha)
    print('config: {}\n'.format(config))
    print('whole test acc ({} tries): {}'.format(config['n_tri'], test_acces))
    print('\tave={:.3f} max={:.3f} min={:.3f}' \
              .format(np.mean(test_acces), np.max(test_acces), np.min(test_acces)))
    
    best_epoch = np.argmax(test_acces)
    alpha = alphas[best_epoch]
    np.save('./result/{}_JKlstm_{}_layerwise_notsort.npy'.format(config['dataset'], config['att_mode']), alpha.to('cpu').detach().numpy().copy())
    


if __name__ == "__main__":
    load()
    main()
