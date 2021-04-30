import itertools
import random
import networkx as nx
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import argparse

import torch
import torch.nn.functional as F
from torch_geometric import utils
from torch_geometric.utils.convert import to_networkx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    return correct.sum() / len(labels), correct

class DictProcessor(argparse.Action):
    def __call__(self, parser, namespace, values, option_strings=None):
        param_dict = getattr(namespace,self.dest,[])
        if param_dict is None:
            param_dict = {}

        k, v = values.split("=")
        param_dict[k] = v
        setattr(namespace, self.dest, param_dict)

class HomophilyRank:
    def __init__(self, mode='avg'):
        self.mode = mode

    def calc_length_of_all_pairs(self, G, n_nodes):
        paths = torch.zeros(n_nodes, n_nodes)

        longest_path_length = 8
        for i in range(n_nodes):
            for j in range(n_nodes):
                try:
                    paths[i][j] = nx.shortest_path_length(G, source=i, target=j)
                except: # there is no path from vi to vj
                    paths[i][j] = longest_path_length + 1
        return paths

    def __call__(self, data):

        # make sub graph
        G = to_networkx(data)

        n_class, n_nodes = torch.max(data.y).data.item() + 1, data.x.size()[0]
        paths = self.calc_length_of_all_pairs(G, n_nodes)
        nodes_idxes_of_class = [torch.where(data.y==c)[0] for c in range(n_class)]
        nodes_idxes_of_ne_class = [torch.where(data.y!=c)[0] for c in range(n_class)]
    
        # add global homophility score ranking
        scores = torch.zeros(n_nodes)
        for vi in range(n_nodes):
            class_of_vi = data.y[vi]
            neighbors  = nodes_idxes_of_class[class_of_vi]
            neighbors_ = nodes_idxes_of_ne_class[class_of_vi]
            avg_dist  = torch.mean(paths[vi][neighbors]) 
            avg_dist_ = torch.mean(paths[vi][neighbors_])
            
            if(self.mode=='avg'):
                scores[vi] = avg_dist_ / avg_dist
            elif(self.mode=='homo'):
                scores[vi] = -avg_dist
            else: # if self.mode is 'hetero'
                scores[vi] = avg_dist_
        data.homophily_rank = torch.argsort(-scores)
        
        return data