import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.construct import rand
import seaborn as sns
import os
import re
from sklearn.manifold import TSNE
import scipy as sp
import sklearn.base
import bhtsne
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, PPI, Reddit
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_networkx, homophily
from torch_geometric.data import NeighborSampler
from torch_geometric.data import Data


def calc_homo(data, name, mode):
    def calc_length_of_all_pairs(G, n_nodes):
        paths = torch.zeros(n_nodes, n_nodes)

        for i in tqdm(range(n_nodes)):
            for j in range(n_nodes):
                try:
                    paths[i][j] = nx.shortest_path_length(
                        G, source=i, target=j)
                except:  # there is no path from vi to vj
                    paths[i][j] = -1
        return paths

    G = to_networkx(data)
    longest_path_length = 6
    n_class, n_nodes = torch.max(data.y), data.x.size()[0]
    if mode == 'multi':
        n_class = data.y.size(1)
    
    if os.path.exists('./result/homophily_score/{}_paths.npy'.format(name)):
        print('paths have been already made')
        paths_np = np.load('./result/homophily_score/{}_paths.npy'.format(name))
        paths = torch.from_numpy(paths_np.astype(np.int16)).clone()
    else:
        paths = calc_length_of_all_pairs(G, n_nodes)
        np.save('./result/homophily_score/{}_paths.npy'.format(name), paths.to('cpu').detach().numpy().copy()) 

    homo_scores = []
    for l in range(1, longest_path_length+1):
        print('{}-th layer'.format(l))
        homo_scores_l_th_layer = torch.zeros(n_nodes)
        for i in tqdm(range(n_nodes)):
            ox_l_from_vi = torch.where(paths[i]==l)[0].tolist() # <= or ==
            # ox_l_from_vi.remove(i)
            if(len(ox_l_from_vi) == 0):
                homo_scores_l_th_layer[i] = 0.
            else:
                if mode == 'single':
                    o_l_from_vi  = torch.where(data.y[ox_l_from_vi]==data.y[i])[0].tolist()
                    homo_scores_l_th_layer[i] = float(len(o_l_from_vi)/len(ox_l_from_vi))
                else: # multi
                    neighbor_size = len(ox_l_from_vi)
                    for n in ox_l_from_vi:
                        o = len(torch.where((data.y[i]==1)*(data.y[n]==1) == True)[0].tolist())
                        homo_scores_l_th_layer[i] += o
                    homo_scores_l_th_layer[i] /= (n_class * neighbor_size)

        homo_scores.append(homo_scores_l_th_layer)
    homo_scores = torch.stack(homo_scores, dim=0)
    
    return homo_scores


data_name = 'PPI' # candidate is [Cora, CiteSeer, PubMed, PPI, Reddit]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root = '../data/{}_None'.format(data_name).lower()
print(data_name)


if data_name in ['Cora', 'CiteSeer', 'PubMed']:
    dataset = Planetoid(root          = root,
                        name          = data_name,
                        split         = 'public')
    data = dataset[0].to(device)

    score = calc_homo(data, data_name, 'single')
    np.savetxt('./result/homophily_score/{}_homo_score.csv'.format(data_name), score.to('cpu').detach().numpy().copy())


elif data_name == 'PPI':
    train_dataset = PPI(root, split='train')
    val_dataset   = PPI(root, split='val')
    test_dataset  = PPI(root, split='test')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    for i, data in enumerate(train_loader):
        data = data.to('cuda')
        graph_name = '{}_train_g{}'.format(data_name, i)
        score = calc_homo(data, graph_name, 'multi')
        np.savetxt('./result/homophily_score/{}_homo_score.csv'.format(graph_name), score.to('cpu').detach().numpy().copy())
    for i, data in enumerate(val_loader):
        data = data.to('cuda')
        graph_name = '{}_val_g{}'.format(data_name, i)
        score = calc_homo(data, graph_name, 'multi')
        np.savetxt('./result/homophily_score/{}_homo_score.csv'.format(graph_name), score.to('cpu').detach().numpy().copy())
    for i, data in enumerate(test_loader):
        data = data.to('cuda')
        graph_name = '{}_test_g{}'.format(data_name, i)
        score = calc_homo(data, graph_name, 'multi')
        np.savetxt('./result/homophily_score/{}_homo_score.csv'.format(graph_name), score.to('cpu').detach().numpy().copy())


elif data_name == 'Reddit':
    dataset = Reddit(root=root)
    data = dataset[0]

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    sizes_l = [15, 10, 5, 5, 3, 3] # i-th elements means aggr size of i-th layer
    train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask,
                                   sizes=sizes_l[:4], batch_size=1024, shuffle=True,
                                   num_workers=3) # sizes is sampling size when aggregates
    test_loader  = NeighborSampler(data.edge_index, node_idx=data.test_mask,
                                   sizes=sizes_l[:4], batch_size=1024, shuffle=False,
                                   num_workers=3) # all nodes is considered

    scores = []
    for bi, (batch_size, n_id, adjs) in enumerate(train_loader):
        y = data.x[n_id].to(device)
        edge_index = torch.cat([adj[0].to(device) for adj in adjs], dim=-1)
        score = homophily.homophily_ratio(edge_index, y)
        print('{}-th train batch: {}'.format(bi, score))
        scores.append(score)
    for bi, (batch_size, n_id, adjs) in enumerate(test_loader):
        y = data.y[n_id].to(device)
        edge_index = torch.cat([adj[0].to(device) for adj in adjs], dim=-1)
        score = homophily.homophily_ratio(edge_index, y)
        print('{}-th test batch: {}'.format(bi, score))
        scores.append(score)
    print('avg score: {}'.format(sum(scores) / len(scores)))