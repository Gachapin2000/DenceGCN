import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torchviz import make_dot
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, to_dense_adj

from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from layers import AttentionSummarize, GeneralConv


class AttGNN_GCNConv(nn.Module):

    def __init__(self, cfg):
        super(AttGNN_GCNConv, self).__init__()
        self.dropout = cfg.dropout

        self.convs = nn.ModuleList()
        self.convs.append(GeneralConv(cfg.task, 'gcn_conv', cfg.n_feat, cfg.n_hid, cfg.self_node, cfg.norm))
        for _ in range(1, cfg.n_layer):
            self.convs.append(GeneralConv(cfg.task, 'gcn_conv', cfg.n_hid, cfg.n_hid, cfg.self_node, cfg.norm))

        self.att = AttentionSummarize(summary_mode = cfg.summary_mode,
                                      att_mode     = cfg.att_mode, 
                                      channels     = cfg.n_hid, 
                                      num_layers   = cfg.n_layer, 
                                      temparature  = cfg.att_temparature,
                                      learn_t      = cfg.learn_temparature)
        self.out_lin = nn.Linear(cfg.n_hid, cfg.n_class)

    def forward(self, x, edge_index):
        hs = []
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.dropout(F.relu(x), self.dropout, training=self.training)
            hs.append(x)

        h, alpha = self.att(hs)  # hs = [h^1,h^2,...,h^L], each h^l is (n, d).
        return self.out_lin(h), alpha


class AttGNN_SAGEConv(nn.Module):
    def __init__(self, cfg):
        super(AttGNN_SAGEConv, self).__init__()
        self.dropout = cfg.dropout
        self.n_layer = cfg.n_layer

        self.convs = nn.ModuleList()
        self.convs.append(GeneralConv(cfg.task, 'sage_conv', cfg.n_feat, cfg.n_hid, cfg.self_node))
        for _ in range(1, cfg.n_layer):
            self.convs.append(GeneralConv(cfg.task, 'sage_conv', cfg.n_hid, cfg.n_hid, cfg.self_node))

        self.att = AttentionSummarize(summary_mode = cfg.summary_mode,
                                      att_mode     = cfg.att_mode, 
                                      channels     = cfg.n_hid, 
                                      num_layers   = cfg.n_layer, 
                                      temparature  = cfg.att_temparature,
                                      learn_t      = cfg.learn_temparature)
        self.out_lin = nn.Linear(cfg.n_hid, cfg.n_class)

    def forward(self, x, adjs, batch_size):
        xs = []
        for l, (edge_index, _, size) in enumerate(adjs): # size is [B_l's size, B_(l+1)'s size]
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[l]((x, x_target), edge_index) # x's shape is (B_l's size, hid) -> (B_(l+1)'s size, hid)
            if l != self.n_layer - 1: # if not the last layer
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)
        xs = [x[:batch_size] for x in xs]

        h, alpha = self.att(xs) # xs = [h^1,h^2,...,h^L], each h^l is (n, d)
        return self.out_lin(h), alpha

    def inference(self, x_all, all_subgraph_loader):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        x_alls = []
        for l in range(self.n_layer):
            xs = []
            for batch_size, n_id, adj in all_subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[l]((x, x_target), edge_index)
                if l != self.n_layer - 1: 
                    x = F.relu(x)
                xs.append(x)

            x_all = torch.cat(xs, dim=0)
            x_alls.append(x_all)

        h, alpha = self.att(x_alls)  # hs = [h1,h2,h3,...,hL], h is (n, d)
        return self.out_lin(h), alpha


class AttGNN_GATConv(nn.Module):

    def __init__(self, cfg):
        super(AttGNN_GATConv, self).__init__()
        self.dropout = cfg.dropout
    
        self.convs = torch.nn.ModuleList()
        in_conv = GeneralConv(cfg.task, 'gat_conv', cfg.n_feat, cfg.n_hid, cfg.self_node, 
                              n_heads=[1, cfg.n_head],
                              iscat=[False, cfg.iscat],
                              dropout=self.dropout)
        self.convs.append(in_conv)
        for _ in range(1, cfg.n_layer):
            conv = GeneralConv(cfg.task, 'gat_conv', cfg.n_hid, cfg.n_hid, cfg.self_node, 
                               n_heads=[cfg.n_head, cfg.n_head],
                               iscat=[cfg.iscat, cfg.iscat],
                               dropout=self.dropout)
            self.convs.append(conv)

        self.att = AttentionSummarize(summary_mode = cfg.summary_mode,
                                      att_mode     = cfg.att_mode, 
                                      channels     = cfg.n_hid * cfg.n_head,
                                      num_layers   = cfg.n_layer, 
                                      temparature  = cfg.att_temparature,
                                      learn_t      = cfg.learn_temparature)
        self.out_lin = nn.Linear(cfg.n_hid * cfg.n_head, cfg.n_class)

    def forward(self, x, edge_index):
        hs = []
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.dropout(F.relu(x), self.dropout, training=self.training)
            hs.append(x)

        h, alpha = self.att(hs)  # hs = [h^1,h^2,...,h^L], each h^l is (n, d)
        return self.out_lin(h), alpha 


class GATNet(nn.Module):
    def __init__(self, task, n_feat, n_hid, n_class, dropout, n_head, iscat):
        super(GATNet, self).__init__()
        self.dropout = dropout

        n_layers = [n_feat] + list(n_hid) + [n_class]
        self.convs = torch.nn.ModuleList()
        for idx in range(len(n_layers)-1):
            conv = GeneralConv(task, 'gat_conv', n_layers[idx], n_layers[idx+1],
                               n_heads=[n_head[idx], n_head[idx+1]],
                               iscat=[iscat[idx], iscat[idx+1]],
                               dropout=self.dropout)
            self.convs.append(conv)
        print(self.convs)

    def forward(self, x, edge_index):
        # atts, es = [], []
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, edge_index)
            if(i < len(self.convs)-1):  # skips elu activate iff last layer
                x = F.elu(x)
            # atts.append(alpha)
            # es.append(edge_index_)
        return x


class GCN(nn.Module):
    def __init__(self, task, n_feat, n_hid, n_class, dropout):
        super(GCN, self).__init__()
        self.dropout = dropout

        n_layers = [n_feat] + list(n_hid) + [n_class]
        self.convs = torch.nn.ModuleList()
        for idx in range(len(n_layers)-1):
            conv = GeneralConv(
                task, 'gcn_conv', n_layers[idx], n_layers[idx+1])
            self.convs.append(conv)
        print(self.convs)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if(i < len(self.convs)-1):  # skips relu activate and dropout iff last layer
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        return x


def return_net(cfg):
    if cfg.model == 'GCN':
        return GCN(task=cfg['task'],
                   n_feat=cfg['n_feat'],
                   n_hid=cfg['n_hid'],
                   n_class=cfg['n_class'],
                   dropout=cfg['dropout'])

    elif cfg.model == 'GATNet':
        return GATNet(task=cfg['task'],
                      n_feat=cfg['n_feat'],
                      n_hid=cfg['n_hid'],
                      n_class=cfg['n_class'],
                      dropout=cfg['dropout'],
                      n_head=cfg['n_head'],
                      iscat=cfg['iscat'])

    elif cfg.model == 'AttGNN':
        if cfg.base_gnn == 'GCN':
            return AttGNN_GCNConv(cfg)
        elif cfg.base_gnn == 'SAGE':
            return AttGNN_SAGEConv(cfg)
        elif cfg.base_gnn == 'GAT':
            return AttGNN_GATConv(cfg)
