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
from layers import JumpingKnowledge, GeneralConv


class JKNet_SAGEConv(nn.Module):
    def __init__(self, task, n_feat, n_hid, n_layer, n_class,
                 dropout, self_node, mode, att_mode, att_temparature):
        super(JKNet_SAGEConv, self).__init__()
        self.dropout = dropout
        self.n_layer = n_layer
        # GeneralConv(task, 'gcn_conv', n_feat, n_hid)
        self.convs = nn.ModuleList()
        self.convs.append(GeneralConv(task, 'sage_conv', n_feat, n_hid, self_node))
        for _ in range(1, n_layer):
            self.convs.append(GeneralConv(task, 'sage_conv', n_feat, n_hid, self_node))

        if(mode == 'attention'):
            self.jk = JumpingKnowledge(
                'attention', att_mode, channels=n_hid, num_layers=n_layer, temparature=att_temparature)
        else:  # if mode == 'cat' or 'max'
            self.jk = JumpingKnowledge(mode)

        if mode == 'cat':
            self.out_lin = nn.Linear(n_hid*n_layer, n_class)
        else:  # if mode == 'max' or 'attention'
            self.out_lin = nn.Linear(n_hid, n_class)

    def forward(self, x, adjs, batch_size):
        xs = []
        for i, (edge_index, _, size) in enumerate(adjs):
            # size is [106991, 21790], may be (B0's size, B1's size)
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            # x is (107741, 602), x_target is (22011, 602) -> x is (22011, 256) (i=0)
            # x is (22011, 256), x_target is (1024, 256) -> x is (1024, 41) (i=1)
            if i != self.n_layer - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)
        xs = [x[:batch_size] for x in xs]

        h, alpha = self.jk(xs) # xs = [h1,h2,h3, ...,hL], h is (n, d)
        return self.out_lin(h), alpha

    def inference(self, x_all, all_subgraph_loader):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        x_alls = []
        for i in range(self.n_layer): # l1, l2
            xs = []
            for batch_size, n_id, adj in all_subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.n_layer - 1:
                    x = F.relu(x)
                xs.append(x)

            x_all = torch.cat(xs, dim=0)
            x_alls.append(x_all)

        h, alpha = self.jk(x_alls)  # xs = [h1,h2,h3,...,hL], h is (n, d)
        return self.out_lin(h), alpha


class JKNet_GCNConv(nn.Module):

    def __init__(self, task, n_feat, n_hid, n_layer, n_class,
                 dropout, self_node, mode, att_mode, att_temparature):
        super(JKNet_GCNConv, self).__init__()
        self.dropout = dropout

        self.in_conv = GeneralConv(task, 'gcn_conv', n_feat, n_hid, self_node)
        self.convs = nn.ModuleList()
        for _ in range(1, n_layer):
            self.convs.append(GeneralConv(task, 'gcn_conv', n_hid, n_hid, self_node))

        if(mode == 'attention'):
            self.jk = JumpingKnowledge(
                'attention', att_mode, channels=n_hid, num_layers=n_layer, temparature=att_temparature)
        else:  # if mode == 'cat' or 'max'
            self.jk = JumpingKnowledge(mode)

        if mode == 'cat':
            self.out_lin = nn.Linear(n_hid*n_layer, n_class)
        else:  # if mode == 'max' or 'attention'
            self.out_lin = nn.Linear(n_hid, n_class)

    def forward(self, x, edge_index):
        x = self.in_conv(x, edge_index)
        x = F.dropout(F.relu(x), self.dropout, training=self.training)

        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.dropout(F.relu(x), self.dropout, training=self.training)
            xs.append(x)

        h, alpha = self.jk(xs)  # xs = [h1,h2,h3,...,hL], h is (n, d)
        return self.out_lin(h), alpha


class JKNet_GATConv(nn.Module):

    def __init__(self, task, n_feat, n_hid, n_layer, n_class,
                 dropout, self_node, mode, att_mode, att_temparature, n_head, iscat):
        super(JKNet_GATConv, self).__init__()
        self.dropout = dropout

        self.in_conv = GeneralConv(task, 'gat_conv', n_feat, n_hid, self_node, 
                                   n_heads=[1, n_head],
                                   iscat=[False, iscat],
                                   dropout=self.dropout)
        
        self.convs = torch.nn.ModuleList()
        for _ in range(1, n_layer):
            conv = GeneralConv(task, 'gat_conv', n_hid, n_hid, self_node, 
                               n_heads=[n_head, n_head],
                               iscat=[iscat, iscat],
                               dropout=self.dropout)
            self.convs.append(conv)

        if(mode == 'attention'):
            self.jk = JumpingKnowledge(
                'attention', att_mode, channels=n_hid*n_head, num_layers=n_layer, temparature=att_temparature)
        else:  # if mode == 'cat' or 'max'
            self.jk = JumpingKnowledge(mode)

        if mode == 'cat':
            self.out_lin = nn.Linear(n_hid*n_head*n_layer, n_class)
        else:  # if mode == 'max' or 'attention'
            self.out_lin = nn.Linear(n_hid*n_head, n_class)

    def forward(self, x, edge_index):
        x = self.in_conv(x, edge_index)
        x = F.dropout(F.relu(x), self.dropout, training=self.training)

        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.dropout(F.relu(x), self.dropout, training=self.training)
            xs.append(x)

        h, alpha = self.jk(xs)  # xs = [h1,h2,h3,...,hL], h is (n, d)
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


def return_net(args):
    if args['model'] == 'GCN':
        return GCN(task=args['task'],
                   n_feat=args['n_feat'],
                   n_hid=args['n_hid'],
                   n_class=args['n_class'],
                   dropout=args['dropout'])

    elif args['model'] == 'GATNet':
        return GATNet(task=args['task'],
                      n_feat=args['n_feat'],
                      n_hid=args['n_hid'],
                      n_class=args['n_class'],
                      dropout=args['dropout'],
                      n_head=args['n_head'],
                      iscat=args['iscat'])

    elif args['model'] == 'JKNet_GCNConv':
        return JKNet_GCNConv(task=args['task'],
                             n_feat=args['n_feat'],
                             n_hid=args['n_hid'],
                             n_layer=args['n_layer'],
                             n_class=args['n_class'],
                             dropout=args['dropout'],
                             self_node=args['self_node'],
                             mode=args['jk_mode'],
                             att_mode=args['att_mode'],
                             att_temparature=args['att_temparature'])

    elif args['model'] == 'JKNet_SAGEConv':
        return JKNet_SAGEConv(task=args['task'],
                              n_feat=args['n_feat'],
                              n_hid=args['n_hid'],
                              n_layer=args['n_layer'],
                              n_class=args['n_class'],
                              dropout=args['dropout'],
                              self_node=args['self_node'],
                              mode=args['jk_mode'],
                              att_mode=args['att_mode'],
                              att_temparature=args['att_temparature'])

    elif args['model'] == 'JKNet_GATConv':
        return JKNet_GATConv(task=args['task'],
                             n_feat=args['n_feat'],
                             n_hid=args['n_hid'],
                             n_layer=args['n_layer'],
                             n_class=args['n_class'],
                             dropout=args['dropout'],
                             self_node=args['self_node'],
                             mode=args['jk_mode'],
                             att_mode=args['att_mode'],
                             att_temparature=args['att_temparature'],
                             n_head=args['n_head'],
                             iscat=args['iscat'])
