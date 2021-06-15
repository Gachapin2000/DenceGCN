import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.construct import rand
import seaborn as sns
import os
import re
from sklearn.manifold import TSNE
import scipy as sp
import sklearn.base
import bhtsne

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import PPI
from torch_geometric.data import DataLoader

from debug import calc_homo
from data import Planetoid

'''for file in glob.glob('./result/*.npy'):
    plt.figure()
    att = np.load(file)
    sns.heatmap(att, vmin=0., vmax=1., center=0.5, cmap='Reds')
    plt.savefig(os.path.splitext(file)[0] + '.png')
    plt.close()'''



class BHTSNE(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, dimensions=2, perplexity=30.0, theta=0.5, rand_seed=-1, max_iter=100):
        self.dimensions = dimensions
        self.perplexity = perplexity
        self.theta = theta
        self.rand_seed = rand_seed
        self.max_iter = max_iter
        print(self.max_iter)

    def fit_transform(self, X):
        return bhtsne.tsne(
            X.astype(sp.float64),
            dimensions=self.dimensions,
            perplexity=self.perplexity,
            theta=self.theta,
            rand_seed=self.rand_seed
        )

def t_sne(data, *, dim=2):
    bh = BHTSNE(dimensions=dim, perplexity=30.0, theta=0.5, rand_seed=-1, max_iter=10000)
    tsne = bh.fit_transform(data)
    
    return tsne


if __name__ == "__main__":
    root = './data/PPI_None'
    train_dataset = PPI(root.lower(), split='train')
    val_dataset   = PPI(root.lower(), split='val')
    test_dataset  = PPI(root.lower(), split='test')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    data_loader = [train_loader, val_loader, test_loader]

    homo_scores_all_graphs = []
    for i, data in enumerate(train_loader):
        data = data.to('cuda')
        homo_scores = calc_homo(data, 'train_g{}'.format(i), 'multi')
        homo_scores_all_graphs.append(homo_scores)
    for i, data in enumerate(val_loader):
        data = data.to('cuda')
        homo_scores = calc_homo(data, 'val_g{}'.format(i), 'multi')
        homo_scores_all_graphs.append(homo_scores)
    for i, data in enumerate(test_loader):
        data = data.to('cuda')
        homo_scores = calc_homo(data, 'test_g{}'.format(i), 'multi')
        homo_scores_all_graphs.append(homo_scores)
    homo_scores_all_graphs = torch.cat(homo_scores_all_graphs, dim=0)
    homo_scores_all_graphs = torch.mean(homo_scores_all_graphs, dim=0)
    np.savetxt('./result/homophily_score/PPI_homo_i_th_layer.csv', homo_scores.to('cpu').detach().numpy().copy())
'''
    x_nps, y_nps = [], []
    for batch_id, data in enumerate(train_loader):
        x_np = data.x.to('cpu').detach().numpy().copy()
        x_nps.append(x_np)
        y_np = np.zeros(x_np.shape[0])
        y_nps.append(y_np)
    for data in test_loader:
        x_np = data.x.to('cpu').detach().numpy().copy()
        x_nps.append(x_np)
        y_np = np.ones(x_np.shape[0])
        y_nps.append(y_np)
    x = np.concatenate(x_nps)
    y = np.concatenate(y_nps)

    x_tsne = t_sne(x)
    xmin = x_tsne[:,0].min()
    xmax = x_tsne[:,0].max()
    ymin = x_tsne[:,1].min()
    ymax = x_tsne[:,1].max()

    # id 0 ('r') is train, id 1 ('b') is test
    colors = ['r', 'b']
    colors = [colors[int(i)] for i in y]
    plt.figure(figsize=(16,12))
    plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=colors)
    plt.axis([xmin,xmax,ymin,ymax])
    plt.xlabel("component 0")
    plt.ylabel("component 1")
    plt.title("PPI t-SNE visualization (blue is test, red is train)")
    plt.savefig("ppi_tsne2.png")
'''


'''if __name__ == "__main__":
    dataset = Planetoid(root='.data/PubMed_None'.lower(), name='PubMed', seed=0, split='public')
    data = dataset[0].to('cuda')

    homo_scores = calc_homo(data)
    np.savetxt('PubMed_homo_i_th_layer.csv', homo_scores.to('cpu').detach().numpy().copy())
    
    colors = []
    for i in range(data.x.size()[0]):
        if(data.test_mask[i]):
            colors.append('b')
        else:
            colors.append('r')

    x = data.x.to('cpu').detach().numpy().copy()
    x = x[::10]
    colors = colors[::10]
    x_tsne = t_sne(x)
    xmin = x_tsne[:,0].min()
    xmax = x_tsne[:,0].max()
    ymin = x_tsne[:,1].min()
    ymax = x_tsne[:,1].max()

    plt.figure(figsize=(16,12))
    plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=colors)
    plt.axis([xmin,xmax,ymin,ymax])
    plt.xlabel("component 0")
    plt.ylabel("component 1")
    plt.title("PubMed t-SNE visualization (blue is test, red is train)")
    plt.savefig("pubmed_tsne.png")'''