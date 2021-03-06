import itertools
from logging import config
import random
import networkx as nx
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from omegaconf import DictConfig, ListConfig
import mlflow

import torch
import torch.nn.functional as F
from torch_geometric import utils
from torch_geometric.utils.convert import to_networkx

def log_params_from_omegaconf_dict(params):
    for param_name, element in params.items():
        mlflow.log_param(param_name, element)
    
def save_conf(config, file):
    with open(file, 'w') as w:
        for key in config.keys():
            w.write('{}\t{}\t{}\n'.format(key, config[key], type(config[key]).__name__))

def read_conf(file):
    config = {}
    with open(file, 'r') as r:
        for line in r.readlines():
            key, value, type_ = line.rstrip().split('\t')
            if   type_=='int': value = int(value)
            elif type_=='float': value = float(value)
            else: value = str(value)
            config[key] = value
    return config

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    return correct.sum() / len(labels), correct


def summarize_acc(model, data, mode='single'):
    model.eval()
    h, _ = model(data.x, data.edge_index)
    if mode == 'single':
        prob_labels = F.log_softmax(h, dim=1)
        preds = prob_labels.max(1)[1].type_as(data.y)
        correct = preds.eq(data.y).double()
    else: # mode == 'multi'
        ys, preds = [], []
        for data in loader: # only one graph (=g1+g2)
            data = data.to(device)
            ys.append(data.y)
            out, alpha = model(data.x, data.edge_index)
            alphas.append(alpha)
            preds.append((out > 0).float().cpu())

    return correct

class HomophilyRank:
    def calc_length_of_all_pairs(self, G, n_nodes):
        paths = torch.zeros(n_nodes, n_nodes)

        longest_path_length = 5
        for i in tqdm(range(n_nodes)):
            for j in range(n_nodes):
                try:
                    paths[i][j] = nx.shortest_path_length(
                        G, source=i, target=j)
                except:  # there is no path from vi to vj
                    paths[i][j] = longest_path_length + 1
        return paths

    def __call__(self, data):
        G = to_networkx(data)

        n_class, n_nodes = torch.max(data.y).data.item() + 1, data.x.size()[0]
        paths = self.calc_length_of_all_pairs(G, n_nodes)
        nodes_idxes_of_class = [torch.where(
            data.y == c)[0] for c in range(n_class)]
        nodes_idxes_of_ne_class = [torch.where(
            data.y != c)[0] for c in range(n_class)]

        # add global homophility score ranking
        score_avg, score_homo, score_hetero = torch.zeros(
            n_nodes), torch.zeros(n_nodes), torch.zeros(n_nodes)
        for vi in range(n_nodes):
            class_of_vi = data.y[vi]
            neighbors = nodes_idxes_of_class[class_of_vi]
            neighbors_ = nodes_idxes_of_ne_class[class_of_vi]
            avg_dist = torch.mean(paths[vi][neighbors])
            avg_dist_ = torch.mean(paths[vi][neighbors_])

            score_avg[vi] = avg_dist_ / avg_dist
            score_homo[vi] = avg_dist
            score_hetero[vi] = avg_dist_
        data.homophily_score = {'avg': score_avg,
                                'homo': score_homo,
                                'hetero': score_hetero}
        data.homophily_rank = {'avg': torch.argsort(-score_avg),
                               'homo': torch.argsort(score_homo),
                               'hetero': torch.argsort(-score_hetero), }

        return data



class HomophilyRank2:
    def __init__(self, border):
        self.border = border

    def calc_length_of_all_pairs(self, G, n_nodes):
        paths = torch.zeros(n_nodes, n_nodes)

        self.longest_path_length = 3
        for i in tqdm(range(n_nodes)):
            for j in range(n_nodes):
                try:
                    paths[i][j] = nx.shortest_path_length(
                        G, source=i, target=j)
                except:  # there is no path from vi to vj
                    paths[i][j] = self.longest_path_length + 1
        return paths

    def __call__(self, data):
        G = to_networkx(data)

        n_class, n_nodes = torch.max(data.y).data.item() + 1, data.x.size()[0]
        paths = self.calc_length_of_all_pairs(G, n_nodes)
        
        scores = torch.zeros(n_nodes)
        for i in tqdm(range(n_nodes)):
            c_i = data.y[i]
            for l in range(1, self.longest_path_length):
                Vox_l = torch.where(paths[i]<=l)[0].tolist()
                Vox_l.remove(i)
                if(len(Vox_l) == 0):
                    l = self.longest_path_length + 1
                    k = 0.99
                    break
                Vo_l  = torch.where(data.y[Vox_l]==c_i)[0].tolist()
                k = float(len(Vo_l)/len(Vox_l))
                
                if k < self.border:
                    break
            scores[i] = l + k

        data.homophily_score = scores
        data.homophily_rank  = torch.argsort(scores)

        return data