import networkx as nx
import numpy as np
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt

import torch
from torch_geometric import utils
from torch_geometric.utils import to_dense_adj, to_networkx
from torch_geometric.datasets import Planetoid, KarateClub


class visualize_gat:
    def __init__(self, atts, es, data, target):
        self.atts = atts
        self.es = es
        self.data = data
        self.n_layers = len(atts)
        self.adj = to_dense_adj(data.edge_index)[0]

        self.highlights = {}
        self.dfs(path=[target], depth=0)

    def calc_accum_att(self, path):
        path_len = len(path)-1
        accum_att = 1.
        for l in range(path_len):
            u = torch.where(self.es[l][0]==path[-(l+1)])[0].tolist()
            v = torch.where(self.es[l][1]==path[-(l+2)])[0].tolist()
            try:
                idx = list(set(u) & set(v))[0]
                accum_att *= self.atts[l][idx]
            except IndexError:
                print('path: {}, l: {}'.format(path, l))
        return accum_att

    def dfs(self, path, depth):
        depth += 1
        if(depth > self.n_layers): return 0
        neighbors = torch.where(self.adj[path[-1]]==1)[0].tolist()
        for neighbor in neighbors:
            path += [neighbor]
            s, t = path[-1], path[0]
            att = self.calc_accum_att(path)
            self.highlights[(s,t)] = att
            self.dfs(path, depth)
            path.pop()

    def visualize(self):
        G = to_networkx(self.data, to_undirected=True)
        edge_width, edge_color = [], []
        for (u,v,_) in G.edges(data=True):
            G.edges[u,v]['weight'] = 1.
            G.edges[u,v]['color'] = 'blue'
        for (u,v), att in self.highlights.items():
            if G.has_edge(u, v):
                G.edges[u,v]['weight'] = 100.*att
            else:
                G.add_edge(u, v, weight=100*att)
            G.edges[u,v]['color'] = 'red'
        edge_width = [d['weight'] for (u,v,d) in G.edges(data=True)]
        edge_color = [d['color'] for (u,v,d) in G.edges(data=True)]
        
        pos = {0 : (1.5, -1),
               1 : (-1, 0.6),
               2 : (0, 0.6),
               3 : (-1.5, 0),
               4 : (-0.5, 0),
               5 : (0.5, 0),
               6 : (-1, -0.6),
               7 : (-1.2, -1),
               8 : (-1.7, -1),
               9 : (0, -0.6),
               10: (2.5, 0),
               11: (3, 0.6),
               12: (4, 0.6),
               13: (3.5, 0),
               14: (4.5, 0),
               15: (3, -0.6),
               16: (4, -0.6),
               17: (4.2, -1),
               18: (4.7, -1),
              }

        label_colors = ['salmon' if self.data.y[idx]==0 else 'skyblue'
                        for idx in range(G.number_of_nodes())]
        nx.draw_networkx_nodes(G, pos=pos, node_color=label_colors)
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edges(G, pos, edge_color=edge_color, width=edge_width)
        plt.savefig('visualize.png')