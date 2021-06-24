import os.path as osp
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


def train(epoch, config, data, train_loader, model, optimizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()

    alphas = []
    total_loss = total_correct = 0
    for batch_size, n_id, adjs in train_loader:
        # batch_size is 1024, 
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs] # 2 adj, because of 2 layer-conv

        optimizer.zero_grad()
        # n_id is (107741)(=[v_53030, v_182890, ...]) , it is Batch_0
        h, alpha = model(data.x[n_id], adjs, batch_size) # out is (1024, 41)
        alphas.append(alpha)
        prob_labels = F.log_softmax(h, dim=1)
        loss = F.nll_loss(prob_labels, data.y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()
    
    return torch.cat(alphas, dim=0)


@torch.no_grad()
def test(config, data, test_loader, model, optimizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    alphas = []
    total_correct = 0
    for batch_size, n_id, adjs in test_loader:
        # batch_size is 1024, 
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs] # 2 adj, because of 2 layer-conv

        # n_id is (107741)(=[v_53030, v_182890, ...]) , it is Batch_0
        h, alpha = model(data.x[n_id], adjs, batch_size) # out is (1024, 41)
        alphas.append(alpha)
        prob_labels = F.log_softmax(h, dim=1)
        total_correct += int(prob_labels.argmax(dim=-1).eq(data.y[n_id[:batch_size]]).sum())

    approx_acc = total_correct / int(data.test_mask.sum())
    
    return approx_acc, torch.cat(alphas, dim=0)

def run(tri, config, data, train_loader, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = return_net(config).to(device)
    optimizer = torch.optim.Adam(params       = model.parameters(), 
                                 lr           = config['learning_rate'], 
                                 weight_decay = config['weight_decay'])
    
    for epoch in tqdm(range(1, config['epochs'])):
        alpha_train = train(epoch, config, data, train_loader, model, optimizer)
    test_acc, alpha_test = test(config, data, test_loader, model, optimizer)

    torch.save(model, './model.pth')
    return test_acc, alpha_train, alpha_test


@hydra.main(config_path='conf', config_name='config')
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
    print(sizes_l[:config['n_layer']])
    train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask,
                                   sizes=sizes_l[:config['n_layer']], batch_size=1024, shuffle=False,
                                   num_workers=12) # sizes is sampling size when aggregates
    test_loader  = NeighborSampler(data.edge_index, node_idx=data.test_mask,
                                   sizes=sizes_l[:config['n_layer']], batch_size=1024, shuffle=False,
                                   num_workers=12) # all nodes is considered

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_acces = np.zeros(config['n_tri'])
    alphas = []
    for tri in range(config['n_tri']):
        test_acc, alpha_train, alpha_test = run(tri, config, data, train_loader, test_loader)
        test_acces[tri] = test_acc
        alphas.append(torch.cat([alpha_train, alpha_test], dim=0))
    print('config: {}\n'.format(config))
    print('whole test acc ({} tries): {}'.format(config['n_tri'], test_acces))
    print('\tave={:.3f} max={:.3f} min={:.3f}' \
              .format(np.mean(test_acces), np.max(test_acces), np.min(test_acces)))

    '''for tri, alpha in enumerate(alphas):
        np.save('./result/layerwise_att/{}_{}layers_JKlstm_{}_layerwise_att_tri{}.npy'
                .format(config['dataset'], config['n_layer'], config['att_mode'], tri), \
                alpha.to('cpu').detach().numpy().copy())'''

if __name__ == "__main__":
    load()
    main()
