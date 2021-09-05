import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from tqdm import trange, tqdm

from torch.nn import Module, Parameter, Linear, LSTM
from torch.nn import LayerNorm, BatchNorm1d

from torch_geometric.nn import MessagePassing, GATConv, GCNConv, SAGEConv, conv
from torch_geometric.utils import add_self_loops, degree


class GeneralConv(nn.Module):
    def __init__(self, task, conv_name, in_channels, out_channels, self_node, norm,
                 n_heads=[1, 1], iscat=[False, False], dropout=0.):
        super(GeneralConv, self).__init__()
        self.task = task

        if conv_name == 'gcn_conv':
            self.conv = GCNConv(in_channels, out_channels, add_self_loops=self_node)
            if(self.task == 'inductive'):  # if transductive, we dont use linear
                self.lin = nn.Linear(in_channels, out_channels)

        elif conv_name == 'sage_conv':
            self.conv = SAGEConv(in_channels, out_channels)
            if(self.task == 'inductive'):  # if transductive, we dont use linear
                self.lin = nn.Linear(in_channels, out_channels)

        elif conv_name == 'gat_conv':
            if iscat[0]:
                in_channels = in_channels * n_heads[0]
            self.conv = GATConv(in_channels=in_channels,
                                out_channels=out_channels,
                                heads=n_heads[1],
                                concat=iscat[1],
                                dropout=dropout,
                                add_self_loops=self_node)
            if self.task == 'inductive':  # if transductive, we dont use linear
                if iscat[1]:
                    out_channels = out_channels * n_heads[1]
                self.lin = nn.Linear(in_channels, out_channels)
        
        if norm != 'None':
            self.norm = eval(norm + '(out_channels)')


    def forward(self, x, edge_index):
        if self.task == 'transductive':
            x = self.conv(x, edge_index)
        elif self.task == 'inductive':
            x = self.conv(x, edge_index) + self.lin(x)
        if hasattr(self, 'norm'):
            return self.norm(x)
        else:
            return x


class AttentionSummarize(torch.nn.Module):
    def __init__(self, summary_mode, att_mode, channels, num_layers, temparature, learn_t):
        super(AttentionSummarize, self).__init__()
        self.summary_mode = summary_mode
        self.att_mode = att_mode
        if learn_t:
            self.att_temparature = Parameter(torch.Tensor([temparature]), requires_grad=True)
        else:
            self.att_temparature = temparature
        
        if self.summary_mode == 'lstm':
            out_channels_of_bi_lstm = (num_layers * channels) // 2
            self.lstm = LSTM(channels, out_channels_of_bi_lstm,
                             bidirectional=True, batch_first=True)
            self.att = Linear(2*out_channels_of_bi_lstm, 1)

        else: # if self.summary_mode == 'vanilla' or 'roll'
            self.att = Linear(2 * channels, 1)

        self.reset_parameters()

    def reset_parameters(self):
        # self.lstm.reset_parameters()
        self.att.reset_parameters()

    def forward(self, hs):
        assert isinstance(hs, list) or isinstance(hs, tuple)
        h = torch.stack(hs, dim=1)  # h is (n, L, d).

        # summary takes h as input, query and key vector as output
        if self.summary_mode == 'vanilla':
            query = h.clone() # query's l-th row is h_i^l
            n_layers = h.size()[1]
            key = query[:, -1, :].repeat(n_layers, 1, 1).permute(1,0,2) # key's all row is h_i^L

        elif self.summary_mode == 'roll':
            query = h.clone() # query's l-th row is h_i^l
            key = torch.roll(h.clone(), -1, dims=1) # key's l-th row is h_i^(l+1)
            query, key, h = query[:, :-1, :], key[:, :-1, :], h[:, :-1, :]

        elif self.summary_mode == 'lstm':
            alpha, _ = self.lstm(h) # alpha (n, L, dL). dL/2 is hid_channels of forward or backward LSTM
            out_channels = alpha.size()[-1]
            query, key = alpha[:, :, :out_channels//2], alpha[:, :, out_channels//2:]

        
        # attention takes query and key as input, alpha as output
        if self.att_mode == 'dp':
            alpha = (query * key).sum(dim=-1) / math.sqrt(query.size()[-1])

        elif self.att_mode == 'ad':
            query_key = torch.cat([query, key], dim=-1)
            alpha = self.att(query_key).squeeze(-1)
            
        elif self.att_mode == 'mx': 
            query_key = torch.cat([query, key], dim=-1)
            alpha_ad = self.att(query_key).squeeze(-1)
            alpha = alpha_ad * torch.sigmoid((query * key).sum(dim=-1))

        alpha = torch.softmax(alpha/self.att_temparature, dim=-1)

        return (h * alpha.unsqueeze(-1)).sum(dim=1), alpha # h_i = \sum_{l} h_i^l * alpha_i^l



class Matcher(nn.Module):
    '''
        Matching between a pair of nodes to conduct link prediction.
        Use multi-head attention as matching model.
    '''
    
    def __init__(self, n_hid, n_out, temperature = 0.1):
        super(Matcher, self).__init__()
        self.n_hid          = n_hid
        self.linear    = nn.Linear(n_hid,  n_out)
        self.sqrt_hd     = math.sqrt(n_out)
        self.drop        = nn.Dropout(0.2)
        self.cosine      = nn.CosineSimilarity(dim=1)
        self.cache       = None
        self.temperature = temperature
    def forward(self, x, ty, use_norm = False):
        tx = self.drop(self.linear(x))
        if use_norm:
            return self.cosine(tx, ty) / self.temperature
        else:
            return (tx * ty).sum(dim=-1) / self.sqrt_hd