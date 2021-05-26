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

from torch_geometric.nn import GATConv, GCNConv
from layers import JumpingKnowledge, GeneralConv

import math


class UniqGCN(nn.Module):

    def __init__(self, task, n_feat, n_hid, n_layer, n_class, dropout):
        super(UniqGCN, self).__init__()

        self.in_gc = GCNConv(n_feat, n_hid)
        self.in_drop = nn.Dropout(dropout)

        layers, jks, dropouts = [], [], []
        for i in range(1, n_layer-1):
            jks.append(JumpingKnowledge("lstm", channels=n_hid, num_layers=i))
            layers.append(GCNConv(n_hid, n_hid))
            dropouts.append(nn.Dropout(dropout))
        self.jks = nn.ModuleList(jks)
        self.layers = nn.ModuleList(layers)
        self.dropouts = nn.ModuleList(dropouts)

        self.out_jk = JumpingKnowledge(
            "lstm", channels=n_hid, num_layers=n_layer-1)
        self.out_gc = GCNConv(n_hid, n_class)

    def forward(self, x, edge_index):
        x = self.in_drop(F.relu(self.in_gc(x, edge_index)))

        xs = [x]
        for jk, layer, dropout in zip(self.jks, self.layers, self.dropouts):
            x = jk(xs)
            x = dropout(F.relu(layer(x, edge_index)))
            xs.append(x)

        x = self.out_jk(xs)
        h = self.out_gc(x, edge_index)
        return h


class JKNet_GCNConv(nn.Module):

    def __init__(self, task, n_feat, n_hid, n_layer, n_class,
                 dropout, mode, att_mode):
        super(JKNet_GCNConv, self).__init__()
        self.task = task
        self.dropout = dropout

        self.in_conv = GeneralConv(self.task, 'gcn_conv', n_feat, n_hid)
        self.convs = nn.ModuleList()
        for _ in range(n_layer-1):
            self.convs.append(GeneralConv(self.task, 'gcn_conv', n_hid, n_hid))

        if(mode == 'lstm'):
            self.jk = JumpingKnowledge(
                'lstm', att_mode, channels=n_hid, num_layers=n_layer)
        else:  # if mode == 'cat' or 'max'
            self.jk = JumpingKnowledge(mode)

        if mode == 'cat':
            self.out_lin = nn.Linear(n_hid*n_layer, n_class)
        else:  # if mode == 'max' or 'lstm'
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
                 dropout, mode, att_mode, n_head, iscat):
        super(JKNet_GATConv, self).__init__()
        self.task = task
        self.dropout = dropout

        self.in_conv = GeneralConv(task, 'gat_conv', n_feat, n_hid,
                                   n_heads=[1, n_head],
                                   iscat=[False, iscat],
                                   dropout=self.dropout)
        self.convs = torch.nn.ModuleList()
        for idx in range(1, n_layer):
            conv = GeneralConv(task, 'gat_conv', n_hid, n_hid,
                               n_heads=[n_head, n_head],
                               iscat=[iscat, iscat],
                               dropout=self.dropout)
            self.convs.append(conv)

        if(mode == 'lstm'):
            self.jk = JumpingKnowledge(
                'lstm', att_mode, channels=n_hid*n_head, num_layers=n_layer)
        else:  # if mode == 'cat' or 'max'
            self.jk = JumpingKnowledge(mode)

        if mode == 'cat':
            self.out_lin = nn.Linear(n_hid*n_head*n_layer, n_class)
        else:  # if mode == 'max' or 'lstm'
            self.out_lin = nn.Linear(n_hid*n_head, n_class)

    def forward(self, x, edge_index):
        x = self.in_conv(x, edge_index)
        x = F.dropout(F.relu(x), self.dropout, training=self.training)

        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.dropout(F.relu(x), self.dropout, training=self.training)
            xs.append(x)

        h = self.jk(xs)  # xs = [h1,h2,h3,...,hL], h is (n, d)
        return self.out_lin(h)


class GATNet(nn.Module):
    def __init__(self, task, n_feat, n_hid, n_class, dropout, n_head, iscat):
        super(GATNet, self).__init__()
        self.task = task
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
        self.task = task
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
                             mode=args['jk_mode'],
                             att_mode=args['att_mode'])

    elif args['model'] == 'JKNet_GATConv':
        return JKNet_GATConv(task=args['task'],
                             n_feat=args['n_feat'],
                             n_hid=args['n_hid'],
                             n_layer=args['n_layer'],
                             n_class=args['n_class'],
                             dropout=args['dropout'],
                             mode=args['jk_mode'],
                             att_mode=args['att_mode'],
                             n_head=args['n_head'],
                             iscat=args['iscat'])

    elif args['model'] == 'UniqGCN':
        return UniqGCN(task=args['task'],
                       n_feat=args['n_feat'],
                       n_hid=args['n_hid'],
                       n_layer=args['n_layer'],
                       n_class=args['n_class'],
                       dropout=args['dropout'])