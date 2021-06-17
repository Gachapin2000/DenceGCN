import numpy as np
from tqdm import tqdm

from torch_geometric.datasets import Planetoid, PPI, Reddit
from torch_sparse import SparseTensor
import torch
from torch_geometric.utils import homophily, to_scipy_sparse_matrix, from_scipy_sparse_matrix
import torch_sparse
from tqdm.std import tqdm
from torch_geometric.data import NeighborSampler

data = 'Reddit' # candidate is [Cora, CiteSeer, PubMed, PPI, Reddit]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root = '../data/{}_None'.format(data).lower()
print(data)

if data in ['Cora', 'CiteSeer', 'PubMed']:
    dataset = Planetoid(root          = root,
                        name          = data,
                        split         = 'public')
    data = dataset[0]
    
    n_nodes = data.x.size()[0]
    adj_sp = to_scipy_sparse_matrix(edge_index=data.edge_index, 
                                    num_nodes=n_nodes)

    n_step = 3
    base_adj_sp = adj_sp.copy()
    for i in range(n_step):
        adj_sp *= adj_sp * base_adj_sp
        adj_sp.data[:] = 1
        print(base_adj_sp, end='\n\n')


elif data == 'Reddit':
    dataset = Reddit(root=root)
    data = dataset[0].to(device)

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    sizes_l = [15, 10, 5, 5, 3, 3]
    n_layer = 3
    train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask,
                                   sizes=sizes_l[:n_layer], batch_size=1024, shuffle=True,
                                   num_workers=3) # sizes is sampling size when aggregates
    test_loader  = NeighborSampler(data.edge_index, node_idx=data.test_mask,
                                   sizes=sizes_l[:n_layer], batch_size=1024, shuffle=False,
                                   num_workers=3) # all nodes is considered


    all_batch_score = []
    for batch_size, n_id, adjs in tqdm(train_loader):
        y = data.y[n_id]
        adjs = [adj[0] for adj in adjs]
        edge_index = torch.cat(adjs, dim=-1)
        num_nodes = n_id.size(0)
        adj_sp = to_scipy_sparse_matrix(edge_index=edge_index, num_nodes=num_nodes)

        base_adj_sp = adj_sp.copy()
        scores = np.zeros(n_layer)
        for l in range(n_layer):
            adj_sp = adj_sp * base_adj_sp
            adj_sp.data[:] = 1
            edge_index = from_scipy_sparse_matrix(adj_sp)
            homophily_score = homophily.homophily_ratio(edge_index, y)
            scores[l] = homophily_score
        all_batch_score.append(scores)
    all_batch_score = np.stack(all_batch_score)


'''    for batch_size, n_id, adjs in test_loader:
        y = data.y[n_id]
        adjs = [adj[0] for adj in adjs]
        edge_index = torch.cat(adjs, dim=-1)
        num_nodes = n_id.size(0)
        adj_sp = to_scipy_sparse_matrix(edge_index=edge_index, num_nodes=num_nodes)

        base_adj_sp = adj_sp.copy()
        for l in range(n_layer):
            adj_sp = adj_sp * base_adj_sp
            adj_sp.data[:] = 1
            edge_index = from_scipy_sparse_matrix(adj_sp)
            homophily_score = homophily.homophily_ratio(edge_index, y)
'''
