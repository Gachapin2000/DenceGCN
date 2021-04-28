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
from torch_geometric.utils import add_self_loops, degree

from torch_geometric.nn import JumpingKnowledge, GATConv
from layers import GCNConv

import math


class JKGCN(nn.Module):
    
    def __init__(self, num_features=1433, num_hiddens=16, num_classes=7, num_layers=6, dropout=0.5, aggr="add", add_self_loop=True, act="ReLU"):

        super().__init__()
        self.num_features = num_features
        self.num_hiddens = num_hiddens
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout = dropout
        self.aggr = aggr
        self.act_name = act
        self.add_self_loop = add_self_loop
        self.act = eval(f"F.{act.lower()}")

        # self.adjuster = nn.Linear(num_features, num_hiddens)
        layers, jks = [], []

        for i in range(1, self.num_layers+1):
            jks.append(JumpingKnowledge("lstm", channels=num_hiddens, num_layers=i))
            layers.append(
                GCNConv(num_hiddens,
                        num_hiddens,
                        add_self_loops=add_self_loop,
                        aggr=aggr))
            
        self.jks = nn.ModuleList(jks)
        self.layers = nn.ModuleList(layers)

        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(self.num_layers)])

        # self.bns = nn.ModuleList([
        #     nn.BatchNorm1d() for _ in range(self.num_layers)])

        # TODO:
        self.jk = JumpingKnowledge("lstm", channels=num_hiddens, num_layers=self.num_layers+1)
        self.lin = nn.Linear(num_hiddens, num_classes)

    def forward(self, x, edge_index):
        # x = self.adjuster(x) # (2708, 1433) -> (2708, 16)
        xs = [x]
        # x = self.gc0(x, edge_index)
        # xs.append(x)

        for jk, layer, dropout in zip(self.jks, self.layers, self.dropouts):
            x = jk(xs)
            x = dropout(self.act(layer(x, edge_index)))
            xs.append(x)

        self.hiddens = [x.cpu().clone().detach() for x in xs]
        # xs is [ h0, h1, h2, ..., hL ]
        h = self.jk(xs)
        # h is (n, d0+d1+d2+...+dL) tensor
        self.hiddens.append(h.cpu().clone().detach())

        h = self.lin(h)
        # h is (n, c) tensor
        self.hiddens.append(h.cpu().clone().detach())
        return F.log_softmax(h, dim=1)


class DenceGCN(nn.Module):

    def __init__(self, num_features=1433, num_hiddens=[16]*7, num_classes=7, num_layers=6, dropout=0.5, aggr="add", add_self_loop=True, act="ReLU"):

        super().__init__()
        self.num_features = num_features
        self.num_hiddens = num_hiddens
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout = dropout
        self.aggr = aggr
        self.act_name = act
        self.add_self_loop = add_self_loop
        self.act = eval(f"F.{act.lower()}")

        num_hiddens = [0] + num_hiddens
        layers = []

        for i in range(1, self.num_layers+1):
            if i == 1:
                num_input = num_features + num_hiddens[i-1]
            else:
                num_input += num_hiddens[i-1]

            layers.append(
                GCNConv(num_input,
                        num_hiddens[i],
                        add_self_loops=add_self_loop,
                        aggr=aggr))

        self.layers = nn.ModuleList(layers)

        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(self.num_layers)])

        # self.bns = nn.ModuleList([
        #     nn.BatchNorm1d() for _ in range(self.num_layers)])

        # TODO:
        self.concat = JumpingKnowledge("cat")
        self.lin = nn.Linear(num_input + num_hiddens[-1], num_classes)

    def forward(self, x, edge_index):

        xs = [x]
        # x = self.gc0(x, edge_index)
        # xs.append(x)

        for layer, dropout in zip(self.layers, self.dropouts):
            x = torch.cat(xs, dim=1)
            x = dropout(self.act(layer(x, edge_index)))
            xs.append(x)

        self.hiddens = [x.cpu().clone().detach() for x in xs]
        # xs is [ h0, h1, h2, ..., hL ]
        h = self.concat(xs)
        # h is (n, d0+d1+d2+...+dL) tensor
        self.hiddens.append(h.cpu().clone().detach())

        h = self.lin(h)
        # h is (n, c) tensor
        self.hiddens.append(h.cpu().clone().detach())
        return F.log_softmax(h, dim=1)

    def __str__(self):

        return "DenseGCN"

    def __repr__(self):

        return f"DenseGCN(num_features={self.num_features}, num_hiddens={self.num_hiddens}, num_classes={self.num_classes}, num_layers={self.num_layers}, dropout={self.dropout}, aggr={self.aggr}, add_self_loop={self.add_self_loop}, act={self.act_name})"


class JKNet(nn.Module):

    def __init__(self, n_feat, n_hid, n_class, dropout, mode):
        super(JKNet, self).__init__()
        self.n_layers = [n_feat] + n_hid
        self.n_conv = len(self.n_layers)-1
        self.dropout = dropout
        
        self.layers = torch.nn.ModuleList()
        self.drops  = torch.nn.ModuleList()
        for idx in range(self.n_conv):
            self.layers.append(GCNConv(self.n_layers[idx], self.n_layers[idx+1], aggr='add'))
            self.drops.append(nn.Dropout(dropout))

        self.jk = JumpingKnowledge(mode)
        if mode == "cat":
            self.lin = nn.Linear(sum(n_hid), n_class)
        elif mode == "max":
            self.lin = nn.Linear(n_hid[-1], n_class)

    def forward(self, x, edge_index):

        xs = []
        for layer, drop in zip(self.layers, self.drops):
            x = layer(x, edge_index)
            x = F.relu(x)
            x = drop(x)
            xs.append(x)

        h = torch.stack(xs, dim=0)
        h = torch.max(h, dim=0)[0]
        h = self.lin(h)
        return F.log_softmax(h, dim=1)


class GATNet(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, dropout, n_head, iscat):
        super(GATNet, self).__init__()
        self.n_layers = [n_feat] + n_hid + [n_class]
        self.n_conv = len(self.n_layers)-1
        self.dropout = dropout

        self.layers = torch.nn.ModuleList()
        for idx in range(self.n_conv):
            if(iscat[idx] == True):
                input_features = n_head[idx] * self.n_layers[idx]
            else:
                input_features = self.n_layers[idx]
                
            self.layers.append(GATConv(in_channels  = input_features, 
                                       out_channels = self.n_layers[idx+1],
                                       heads        = n_head[idx+1], 
                                       concat       = iscat[idx+1], 
                                       dropout      = self.dropout))

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = F.dropout(x, self.dropout, training=self.training)
            x = layer(x, edge_index)
        x = F.elu(x)
        return F.log_softmax(x, dim=1)


class GCN(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, dropout):
        super(GCN, self).__init__()
        self.n_layers = [n_feat] + n_hid + [n_class]
        self.n_conv = len(self.n_layers)-1
        self.dropout = dropout
        
        self.layers = torch.nn.ModuleList()
        for idx in range(self.n_conv):
            self.layers.append(GCNConv(self.n_layers[idx], self.n_layers[idx+1]))

    def forward(self, x, edge_index):
        for layer in self.layers[0:-1]:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

        # iff the last convolutional layer, we don't use relu and dropout
        x = self.layers[-1](x, edge_index)

        return F.log_softmax(x, dim=1)


def return_net(args):
    if args['model'] == 'GCN':
        return GCN(n_feat  = args['n_feat'],
                   n_hid   = args['n_hid'],
                   n_class = args['n_class'],
                   dropout = args['dropout'])

    elif args['model'] == 'GATNet':
        return GATNet(n_feat  = args['n_feat'],
                      n_hid   = args['n_hid'],
                      n_class = args['n_class'],
                      dropout = args['dropout'],
                      n_head = args['n_head'],
                      iscat   = args['iscat'])
    
    elif args['model'] == 'JKNet':
        return JKNet(n_feat  = args['n_feat'],
                     n_hid   = args['n_hid'],
                     n_class = args['n_class'],
                     dropout = args['dropout'],
                     mode    = args['mode'])
    