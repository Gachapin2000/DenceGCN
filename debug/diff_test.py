import argparse
import numpy as np
import matplotlib.pyplot as plt


def calc_kldiv(a_vec, b_vec):
    return np.sum([a * np.log(a/b) for a, b in zip(a_vec, b_vec)])


parser = argparse.ArgumentParser()
parser.add_argument('--acc1_path', type=str, default=None) # base
parser.add_argument('--acc2_path', type=str, default=None) # new
parser.add_argument('--att_path', type=str, default=None) # new
parser.add_argument('--save', type=str, default='test') # new
args = parser.parse_args()

acces1 = np.load(args.acc1_path)
acces2 = np.load(args.acc2_path)
att = np.load(args.att_path)

new_idx = np.array([idx for idx, (acc1, acc2) in enumerate(zip(acces1, acces2)) \
                if acc1==0 and acc2==1])
att_old, att_new = np.delete(att, new_idx, 0), att[new_idx]

n_layers = np.shape(att)[1]
uniform_dist = np.full_like(att[0], 1./n_layers)

n_nodes_old, n_nodes_new = np.shape(att_old)[0], np.shape(att_new)[0]
kl_divs_old, kl_divs_new = [], []
for v_att_y in att_old:
    kl_div = calc_kldiv(v_att_y, uniform_dist)
    kl_divs_old.append(kl_div)
for v_att_y in att_new:
    kl_div = calc_kldiv(v_att_y, uniform_dist)
    kl_divs_new.append(kl_div)

points = (kl_divs_old, kl_divs_new)

fig, ax = plt.subplots()
bp = ax.boxplot(points)
ax.set_xticklabels(['old', 'new'])

plt.title('Box plot')
# plt.xlabel('')
plt.ylabel('each node')
# plt.ylim([0,100])
plt.grid()

plt.savefig('./result/layerwise_att/{}.png'.format(args.save))