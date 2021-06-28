import os
import sys
import argparse
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
import glob

import torch
from torch_geometric.datasets import PPI, Reddit
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.data import NeighborSampler
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models import return_net
from utils import accuracy, read_conf, HomophilyRank
from data import Planetoid


def load(path, tri_id):
    config = read_conf(path + '/config.txt')

    model = return_net(config).to(device)
    model.load_state_dict(torch.load(path + '/{}th_model.pth'.format(tri_id)))
    model.eval()

    root = './data/{}_{}'.format(config['dataset'], config['pre_transform'])
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    if config['dataset'] in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(root          = root.lower(),
                            name          = config['dataset'],
                            seed          = 0,
                            split         = config['split'], 
                            transform     = eval(config['transform']),
                            pre_transform = eval(config['pre_transform']))
        data = dataset[0].to(device)
        
        return config, data, model

    elif config['dataset'] == 'PPI':
        train_dataset = PPI(root.lower(), split='train')
        test_dataset  = PPI(root.lower(), split='test')
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

        return config, [train_loader, test_loader], model

    else: # Reddit
        dataset = Reddit(root=root.lower())
        data = dataset[0].to(device)
        sizes_l = [25, 10, 10, 10, 10, 10]
        train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask,
                                       sizes=sizes_l[:config['n_layer']], batch_size=1024, shuffle=False,
                                       num_workers=12) # sizes is sampling size when aggregates
        test_loader  = NeighborSampler(data.edge_index, node_idx=data.test_mask,
                                       sizes=sizes_l[:config['n_layer']], batch_size=1024, shuffle=False,
                                       num_workers=12) # all nodes is considered
        
        return config, [data, train_loader, test_loader], model


def calc_kldiv(a_vec, b_vec):
    return np.sum([a * np.log(a/b) for a, b in zip(a_vec, b_vec)])


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='./models/PPI_JKNet_GATConv_5layer_lstm_go/')
parser.add_argument('--tri_id', type=int, default=0)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config, data, model = load(args.path, args.tri_id)

if config['dataset'] in ['Cora', 'CiteSeer', 'PubMed']:
    h, att = model(data.x, data.edge_index)
    att_y = np.load('./result/homophily_score/Planetoid/{}_homo_score.npy'.format(config['dataset']))

elif config['dataset'] == 'PPI':
    train_loader, test_loader = data
    atts = []
    for data in train_loader: # in [g1, g2, ..., g20]
        data = data.to(device)
        out, att = model(data.x, data.edge_index)
        atts.append(att)
    ys, preds = [], []
    for data in test_loader: # only one graph (=g1+g2)
        data = data.to(device)
        out, att = model(data.x, data.edge_index)
        atts.append(att)

    homo_file_list = list(glob.glob('./result/homophily_score/PPI/*.npy'))
    atts_y = [np.load(file) for file in natsorted(homo_file_list)]
    att_y = np.concatenate(atts_y)
    att = torch.cat(atts, dim=0)

else: # Reddit
    data, train_loader, test_loader = data
    atts = []
    for batch_size, n_id, adjs in train_loader:
        adjs = [adj.to(device) for adj in adjs]
        h, att = model(data.x[n_id], adjs, batch_size)
        atts.append(att)
    for batch_size, n_id, adjs in test_loader:
        adjs = [adj.to(device) for adj in adjs]
        h, att = model(data.x[n_id], adjs, batch_size)
        atts.append(att)

    homo_file_list = list(glob.glob('./result/homophily_score/Reddit/*.npy'))
    atts_y = [np.load(file) for file in natsorted(homo_file_list)]
    att_y = np.concatenate(atts_y)
    att = torch.cat(atts, dim=0)


att = att.to('cpu').detach().numpy().copy()
att_y = att_y[:, :config['n_layer']]

epsilon_ary = np.full_like(att_y, 1e-5)
att_y += epsilon_ary
rowsum = np.sum(att_y, axis=-1)
rowsum_inv = np.power(rowsum, -1)
att_y = np.array([np.dot(vec, normalize_coefficient)
                    for vec, normalize_coefficient in zip(att_y, rowsum_inv)])

epsilon_vec = np.array([1e-5 for _ in range(config['n_layer'])])
kl_divs = []
for vi_att, vi_att_y in zip(att, att_y):
    vi_att += epsilon_vec
    vi_att_y += epsilon_vec
    kl_div = calc_kldiv(vi_att, vi_att_y)
    kl_divs.append(kl_div)

fig, ax = plt.subplots()

bp = ax.boxplot(kl_divs)
plt.title('KL divergence distribution between homophily and att layerwise')
plt.xlabel('exams')
plt.ylabel('each node')
# plt.ylim([0,100])
plt.grid()

plt.savefig('./result/kldiv_between_homophily_and_layerwise_att/{}_kldiv_bet_homophily_and_{}base_att_{}layer.png'
            .format(config['dataset'], config['att_mode'], config['n_layer']))