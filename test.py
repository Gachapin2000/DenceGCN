import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.construct import rand
import seaborn as sns
import os
from sklearn.manifold import TSNE

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import PPI
from torch_geometric.data import DataLoader

from data import Planetoid

'''for file in glob.glob('./result/*.npy'):
    plt.figure()
    att = np.load(file)
    sns.heatmap(att, vmin=0., vmax=1., center=0.4, cmap='Reds')
    plt.savefig(os.path.splitext(file)[0] + '.png')
    plt.close()
'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root = './data/PPI_None'
train_dataset = PPI(root.lower(), split='train')
val_dataset   = PPI(root.lower(), split='val')
test_dataset  = PPI(root.lower(), split='test')
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
data_loader = [train_loader, val_loader, test_loader]

x_nps, y_nps = [], []

for data in train_loader:
    data = data.to(device)
    x_np = data.x.to('cpu').detach().numpy().copy()
    x_nps.append(x_np)
    y_np = np.zeros(x_np.shape[0])
    y_nps.append(y_np)

for data in test_loader:
    data = data.to(device)
    x_np = data.x.to('cpu').detach().numpy().copy()
    x_nps.append(x_np)
    y_np = np.ones(x_np.shape[0])
    y_nps.append(y_np)

x = np.concatenate(x_nps)
y = np.concatenate(y_nps)
x_t_sne = TSNE(n_components=2, random_state=0).fit_transform(x)
print(x_t_sne)

xmin = x_t_sne[:,0].min()
xmax = x_t_sne[:,0].max()
ymin = x_t_sne[:,1].min()
ymax = x_t_sne[:,1].max()

plt.figure(figsize=(16,12))
plt.scatter(x_t_sne[:, 0], x_t_sne[:, 1], c=y)
plt.axis([xmin,xmax,ymin,ymax])
plt.xlabel("component 0")
plt.ylabel("component 1")
plt.title("PPI t-SNE visualization")
plt.savefig("ppi_tsne.png")
    