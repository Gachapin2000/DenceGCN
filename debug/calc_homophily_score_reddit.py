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
import networkx as nx

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, PPI, Reddit, KarateClub
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_networkx, homophily, subgraph
from torch_geometric.data import NeighborSampler
from torch_geometric.data import Data


def calc_homophily_of_all_nodes(target, y, edge_indexes, max_depth):
    one_hop_neighbors = []
    for edge_index, _, size in reversed(edge_indexes):
        adj = {i: set() for i in range(size[1])}
        for vi, vj in zip(edge_index[0], edge_index[1]):
            adj[vj.item()].add(vi.item())
        one_hop_neighbors.append(adj)

    all_vertex_scores = []
    for vi in target:
        khop_neighbors_of_vi = [set() for _ in range(max_depth+1)]
        dfs_khop_neighbors(vi, 0, max_depth, khop_neighbors_of_vi, one_hop_neighbors)
        all_vertex_scores.append(calc_homophily_of_vi(vi, y, khop_neighbors_of_vi))
    all_vertex_scores = torch.stack(all_vertex_scores, dim=0)
    return all_vertex_scores


def dfs_khop_neighbors(node, depth, max_depth, khop_neighbors, one_hop_neighbors):
    if node in khop_neighbors[depth]:
        return
    else:
        khop_neighbors[depth].add(node)
    
    if depth+1 > max_depth:
        return
    for n in one_hop_neighbors[depth][node]:
        dfs_khop_neighbors(n, depth+1, max_depth, khop_neighbors, one_hop_neighbors)


def calc_homophily_of_vi(vi, y, khop_neighbors_of_vi):
    n_layer = len(khop_neighbors_of_vi)
    homophily_score_layerwise = []
    for l in range(1, n_layer):
        neighbor_size = len(khop_neighbors_of_vi[l])
        if neighbor_size == 0:
            homophily_score_layerwise.append(0.)
        else:
            ids_neighbors = torch.tensor([[n for n in khop_neighbors_of_vi[l]]])
            labels_neighbors = y[ids_neighbors]
            acc = len(torch.where(labels_neighbors == y[vi])[0].tolist()) / neighbor_size
            homophily_score_layerwise.append(acc)
    return torch.tensor(homophily_score_layerwise)


#---------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root = '../data/Reddit_None'.lower()

dataset = Reddit(root=root)
data = dataset[0]
max_depth = 6
sizes_l = [25,10,10,10,10,10]

torch.manual_seed(0)
torch.cuda.manual_seed(0)
train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask,
                               sizes=sizes_l[:max_depth], batch_size=1024, shuffle=False,
                               num_workers=12) # sizes is sampling size when aggregates
test_loader  = NeighborSampler(data.edge_index, node_idx=data.test_mask,
                               sizes=sizes_l[:max_depth], batch_size=1024, shuffle=False,
                               num_workers=12) # all nodes is considered
loader_names = ['train_loader', 'test_loader']

for loader_name in loader_names:
    print(loader_name)
    loader = eval(loader_name)
    for batch_id, (batch_size, n_id, adjs) in tqdm(enumerate(loader)):
        y = data.y[n_id]
        edge_indexes = [adj for adj in adjs]
        target = range(batch_size)
        all_nodes_scores = calc_homophily_of_all_nodes(target, y, edge_indexes, max_depth)
        np.save('./result/homophily_score/Reddit_25aggr/{}_b{}_homo_score.npy'.format(loader_name, batch_id), 
                all_nodes_scores.to('cpu').detach().numpy().copy())
