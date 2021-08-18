import os
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Reddit
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import SAGEConv

from models import return_net
from utils import save_conf

def train(epoch, config, data, train_loader, model, optimizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()

    for batch_size, n_id, adjs in train_loader:
        adjs = [adj.to(device) for adj in adjs]
        optimizer.zero_grad()
        h, _ = model(data.x[n_id], adjs, batch_size)
        prob_labels = F.log_softmax(h, dim=1)
        loss = F.nll_loss(prob_labels, data.y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()


@torch.no_grad()
def test(config, data, test_loader, model, optimizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    '''total_correct = 0
    for batch_size, n_id, adjs in test_loader:
        adjs = [adj.to(device) for adj in adjs]
        h, alpha = model(data.x[n_id], adjs, batch_size)
        prob_labels = F.log_softmax(h, dim=1)
        total_correct += int(prob_labels.argmax(dim=-1).eq(data.y[n_id[:batch_size]]).sum())
    approx_acc = total_correct / int(data.test_mask.sum())'''

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

    return test_acc, model


@hydra.main(config_path='conf', config_name='config')
def load(cfg : DictConfig) -> None:
    global config
    config = cfg[cfg.key]

def main():
    global config
    print(config)
    dir_ = './models/{}_{}_full_{}layer_{}_{}'.format(config['dataset'],config['model'],config['n_layer'],config['jk_mode'],config['att_mode'])
    # dir_ = './models/test_sd'
    os.makedirs(dir_)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = './data/{}_{}'.format(config['dataset'], config['pre_transform'])
    dataset = Reddit(root=root.lower())
    data = dataset[0].to(device)

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    sizes_l = [25, 10, 10, 10, 10, 10]
    train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask,
                                   sizes=sizes_l[:config['n_layer']], batch_size=1024, shuffle=False,
                                   num_workers=6) # sizes is sampling size when aggregates

    all_subgraph_loader = NeighborSampler(data.edge_index, node_idx=None,
                                          sizes=[-1], batch_size=1024, shuffle=False,
                                          num_workers=6) # all nodes is considered

    test_acc = np.zeros(config['n_tri'])
    for tri in range(config['n_tri']):
        test_acc[tri], model = run(tri, config, data, train_loader, all_subgraph_loader)
        print('{} test acc: {}'.format(config['att_mode'] ,test_acc[tri]))
        torch.save(model.state_dict(), dir_ + '/{}th_model.pth'.format(tri))

    save_conf(config, dir_ + '/config.txt')
    with open(dir_ + '/acc.txt', 'w') as w:
        w.write('whole test acc ({} tries) = {}'.format(config['n_tri'], test_acc))
        w.write('\tave={:.3f} max={:.3f} min={:.3f}' \
              .format(np.mean(test_acc), np.max(test_acc), np.min(test_acc)))
        
if __name__ == "__main__":
    load()
    main()
