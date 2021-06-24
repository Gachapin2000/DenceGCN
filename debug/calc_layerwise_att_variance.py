import numpy as np
from tqdm import tqdm
import glob
import argparse
import matplotlib.pyplot as plt
from natsort import natsorted

from torch_geometric.datasets import Planetoid, PPI, Reddit
from torch_sparse import SparseTensor
import torch
from torch_geometric.utils import homophily, to_scipy_sparse_matrix, from_scipy_sparse_matrix
import torch_sparse
from tqdm.std import tqdm
from torch_geometric.data import NeighborSampler

def calc_kldiv(a_vec, b_vec):
    return np.sum([a * np.log(a/b) for a, b in zip(a_vec, b_vec)])

def normalize(ary):
    epsilon_ary = np.full_like(ary, 1e-5)
    ary += epsilon_ary
    rowsum = np.sum(att_y, axis=-1)
    rowsum_inv = np.power(rowsum, -1)
    ary = np.array([np.dot(vec, normalize_coefficient)
                        for vec, normalize_coefficient in zip(ary, rowsum_inv)])
    return ary


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='../result/homophily_score/Planetoid/CiteSeer*.npy')
parser.add_argument('--n_layers', type=int, default=6)
parser.add_argument('--save', type=str, default='test')

args = parser.parse_args()

files = list(glob.glob(args.path))
att_y = [np.load(file) for file in natsorted(files)]
att_y = np.concatenate(att_y)

att_y = att_y[:, :args.n_layers]
att_y = normalize(att_y)

uniform_dist = np.full_like(att_y[0], 1./args.n_layers)

n_nodes = np.shape(att_y)[0]
kl_divs = np.zeros(n_nodes)
for v_id, v_att_y in enumerate(att_y):
    kl_div = calc_kldiv(v_att_y, uniform_dist)
    kl_divs[v_id] = kl_div
print(np.mean(kl_divs))

fig, ax = plt.subplots()

bp = ax.boxplot(kl_divs)
plt.title('KLdiv distribution between homophily and uni-distribution layerwise')
plt.xlabel('exams')
plt.ylabel('each node')
# plt.ylim([0,100])
plt.grid()

plt.savefig('./result/homophily_score/{}.png'.format(args.save))