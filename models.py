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
from layers import JumpingKnowledge

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

        self.out_jk = JumpingKnowledge("lstm", channels=n_hid, num_layers=n_layer-1)
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


class JKNet(nn.Module):

    def __init__(self, task, n_feat, n_hid, n_layer, n_class, dropout, mode, att_mode):
        super(JKNet, self).__init__()
        self.task = task

        self.in_gc   = GCNConv(n_feat, n_hid)
        self.in_lin  = nn.Linear(n_feat, n_hid)
        self.in_drop = nn.Dropout(dropout)
        
        self.convs = nn.ModuleList()
        self.lins  = nn.ModuleList()
        self.drops = nn.ModuleList()
        for idx in range(n_layer-1):
            self.convs.append(GCNConv(n_hid, n_hid))
            self.lins.append(nn.Linear(n_hid, n_hid))
            self.drops.append(nn.Dropout(dropout))

        if(mode == 'lstm'):
            self.jk = JumpingKnowledge('lstm', att_mode, channels=n_hid, num_layers=n_layer)
        else: # if mode == 'cat' or 'max'
            self.jk = JumpingKnowledge(mode)

        if mode == 'cat':
            self.lin = nn.Linear(n_hid*n_layer, n_class)
        else: # if mode == 'max' or 'lstm'
            self.lin = nn.Linear(n_hid, n_class)

    def forward(self, x, edge_index):
        if self.task == 'inductive':
            x = self.in_gc(x, edge_index) + self.in_lin(x)
        elif self.task == 'transductive':
            x = self.in_gc(x, edge_index)        
        x = self.in_drop(F.relu(x))

        xs = [x]
        for layer, lin, drop in zip(self.convs, self.lins, self.drops):
            if self.task == 'inductive':
                x = layer(x, edge_index) + lin(x)
            elif self.task == 'transductive':
                x = layer(x, edge_index)
            x = drop(F.relu(x))
            xs.append(x)

        h = self.jk(xs) # xs = [h1,h2,h3,...,hL], h is (n, d)
        h = self.lin(h)
        return h


class GATNet(nn.Module):
    def __init__(self, task, n_feat, n_hid, n_class, dropout, n_head, iscat):
        super(GATNet, self).__init__()
        self.task = task
        self.dropout = dropout

        n_layers = [n_feat] + list(n_hid) + [n_class]
        self.convs = torch.nn.ModuleList()
        self.lins  = torch.nn.ModuleList()
        for idx in range(len(n_layers)-1):
            if(iscat[idx] == True):
                input_features = n_head[idx] * n_layers[idx]
            else:
                input_features = n_layers[idx]
                
            self.convs.append(GATConv(in_channels  = input_features, 
                                       out_channels = n_layers[idx+1],
                                       heads        = n_head[idx+1], 
                                       concat       = iscat[idx+1], 
                                       dropout      = self.dropout))
            if(iscat[idx+1] == True):
                self.lins.append(torch.nn.Linear(input_features, n_layers[idx+1]*n_head[idx+1]))
            else:
                self.lins.append(torch.nn.Linear(input_features, n_layers[idx+1]))

        print(self.convs)
        print(self.lins)

    def forward(self, x, edge_index):
        atts, es = [], []
        for i, (conv, lin) in enumerate(zip(self.convs, self.lins)):
            x = F.dropout(x, self.dropout, training=self.training)
            if self.task == 'inductive':
                x = conv(x, edge_index) + lin(x)
            elif self.task == 'transductive':
                x = conv(x, edge_index)
            if(i < len(self.convs)-1): # skips elu activate iff last layer
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
        self.lins  = torch.nn.ModuleList()
        for idx in range(len(n_layers)-1):
            self.convs.append(GCNConv(n_layers[idx], n_layers[idx+1]))
            self.lins.append(torch.nn.Linear(n_layers[idx], n_layers[idx+1]))

    def forward(self, x, edge_index):
        for conv, lin in zip(self.convs[0:-1], self.lins[0:-1]):
            if self.task == 'inductive':
                x = conv(x, edge_index) + lin(x)
            elif self.task == 'transductive':
                x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

        # iff the last convolutional layer, we don't use relu and dropout
        if(self.task == 'inductive'):
            return self.convs[-1](x, edge_index) + self.lins[-1](x)
        elif self.task == 'transductive':
            return self.convs[-1](x, edge_index)


def return_net(args):
    if args['model'] == 'GCN':
        return GCN(task    = args['task'],
                   n_feat  = args['n_feat'],
                   n_hid   = args['n_hid'],
                   n_class = args['n_class'],
                   dropout = args['dropout'])

    elif args['model'] == 'GATNet':
        return GATNet(task    = args['task'],
                      n_feat  = args['n_feat'],
                      n_hid   = args['n_hid'],
                      n_class = args['n_class'],
                      dropout = args['dropout'],
                      n_head = args['n_head'],
                      iscat   = args['iscat'])
    
    elif args['model'] == 'JKNet':
        return JKNet(task    = args['task'],
                     n_feat   = args['n_feat'],
                     n_hid    = args['n_hid'],
                     n_layer  = args['n_layer'],
                     n_class  = args['n_class'],
                     dropout  = args['dropout'],
                     mode     = args['jk_mode'],
                     att_mode = args['att_mode'])

    elif args['model'] == 'UniqGCN':
        return UniqGCN(task    = args['task'],
                       n_feat  = args['n_feat'],
                       n_hid   = args['n_hid'],
                       n_layer = args['n_layer'],
                       n_class = args['n_class'],
                       dropout = args['dropout'])