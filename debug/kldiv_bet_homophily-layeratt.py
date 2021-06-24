import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import argparse
import glob
import os
from natsort import natsorted


def calc_kldiv(a_vec, b_vec):
    return np.sum([a * np.log(a/b) for a, b in zip(a_vec, b_vec)])


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='PPI')
parser.add_argument('--n_layers', type=int, default=3)
parser.add_argument('--att_mode', type=str, default='go')
parser.add_argument('--tri_id', type=int, default=0)
args = parser.parse_args()

att   = np.load('./result/layerwise_att/{}_{}layers_JKlstm_{}_layerwise_att_tri{}.npy' \
                .format(args.dataset, args.n_layers, args.att_mode, args.tri_id))

if args.dataset in ['Cora', 'CiteSeer', 'PubMed']: 
    att_y = np.load('./result/homophily_score/Planetoid/{}_homo_score.npy'.format(args.dataset))

else: # args.dataset == 'PPI' or 'Reddit'
    homo_file_list = list(glob.glob('./result/homophily_score/{}/*.npy'.format(args.dataset)))
    att_y = [np.load(file) for file in natsorted(homo_file_list)]
    att_y = np.concatenate(att_y)

att_y = att_y[:, :args.n_layers]
epsilon_ary = np.full_like(att_y, 1e-5)
att_y += epsilon_ary
rowsum = np.sum(att_y, axis=-1)
rowsum_inv = np.power(rowsum, -1)
att_y = np.array([np.dot(vec, normalize_coefficient)
                    for vec, normalize_coefficient in zip(att_y, rowsum_inv)])

epsilon_vec = np.array([1e-5 for _ in range(args.n_layers)])
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

plt.savefig('./result/kldiv_between_homophily_and_layerwise_att/{}_kldiv_bet_homophily_and_{}base_att_{}layer_{}.png'
            .format(args.dataset, args.att_mode, args.n_layers, args.att_mode))