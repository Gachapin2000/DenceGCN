import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf

import torch
from torch_geometric.datasets import Reddit
from torch_geometric.datasets import PPI
from torch_geometric.data import DataLoader
from torch_geometric.data import NeighborSampler
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models import return_net


@hydra.main(config_path='../conf', config_name='config')
def load(cfg : DictConfig) -> None:
    global config
    config = cfg[cfg.key]

def main():
    global config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    root = './data/{}_{}'.format(config['dataset'], config['pre_transform'])
    dataset = Reddit(root=root.lower())
    data = dataset[0].to(device)

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    sizes_l = [25, 10, 10, 10, 10, 10]
    train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask,
                                   sizes=sizes_l[:config['n_layer']], batch_size=1024, shuffle=False,
                                   num_workers=12) # sizes is sampling size when aggregates
    test_loader  = NeighborSampler(data.edge_index, node_idx=data.test_mask,
                                   sizes=sizes_l[:config['n_layer']], batch_size=1024, shuffle=False,
                                   num_workers=12) # all nodes is considered

    model = return_net(config).to(device)
    model.load_state_dict(torch.load('./model.pth'))

    alphas = []
    total_correct = 0
    for batch_size, n_id, adjs in test_loader:
        adjs = [adj.to(device) for adj in adjs]
        h, alpha = model(data.x[n_id], adjs, batch_size)
        alphas.append(alpha)
        prob_labels = F.log_softmax(h, dim=1)
        total_correct += int(prob_labels.argmax(dim=-1).eq(data.y[n_id[:batch_size]]).sum())

    approx_acc = total_correct / int(data.test_mask.sum())

    print(approx_acc)

if __name__ == "__main__":
    load()
    main()