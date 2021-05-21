import os.path as osp
import networkx as nx
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.io import read_planetoid_data
from torch_geometric.utils.convert import to_networkx


class FiveUniqueNodes(InMemoryDataset):
    def __init__(self, root, split="public", x_std=0.1, transform=None, pre_transform=None):
        super(FiveUniqueNodes, self).__init__(root, transform, pre_transform)

        self.n_nodes = 19

        idxes_of_c1 = [1,2,3,4,5,6,7,8,9,12]
        self.y = torch.tensor([0 if idx in idxes_of_c1 else 1 for idx in range(self.n_nodes)])
        self.x = self.make_x(x_std)

        row = torch.LongTensor([1,1,1,2,2,2,3,3,3,4,4,4,4,4,4,5,5,5,5 ,5 ,6,6,6,6,7,7,8,9,9,9,9 ,10,10,10,10,10,11,11,11,12,12,12,13,13,13,13,13,13,14,14,14,15,15,15,15,16,16,16,16,17,17,18])
        col = torch.LongTensor([2,3,4,1,4,5,1,4,6,1,2,3,5,6,9,2,4,9,10,15,3,4,7,9,6,8,7,4,5,6,10,5 ,9 ,11,13,15,10,12,13,11,13,14,10,11,12,14,15,16,12,13,16,5, 10,13,16,13,14,15,17,16,18,17])
        
        edge_index = torch.stack([row, col], dim=0)

        train_mask = torch.zeros(self.n_nodes, dtype=torch.bool)
        if(split=='public'):
            for i in [4,13,16]:
                train_mask[i] = True
        elif(split=='full_100per'):
            train_mask.fill_(True)

        data = Data(x=self.x, edge_index=edge_index, y=self.y, train_mask=train_mask)
        self.data, self.slices = self.collate([data])

        # self.visualize()

    def make_x(self, std):
        n_features = 2
        centers = [0.25, 0.75]
        x = np.zeros((self.n_nodes, n_features))
        for idx in range(self.n_nodes):
            x[idx][0] = np.random.normal(centers[self.y[idx]], std)
            x[idx][1] = np.random.normal(0.5, 0.5)
        return torch.from_numpy(x.astype(np.float32))
        
    def visualize(self):
        # plot x of toy data
        plt.scatter(self.x[:, 0], self.x[:, 1], c=self.y)
        for idx in range(self.x.size()[0]):
            plt.annotate(idx, (self.x[idx][0], self.x[idx][1]))
        plt.savefig('toy_x.png')
        plt.show()

        # plot edge_index of toy data
        G = to_networkx(self.data, to_undirected=True)
        pos = nx.spring_layout(G, k=0.5)
        label_colors = ['salmon' if self.data.y[idx]==0 else 'skyblue'
                        for idx in range(G.number_of_nodes())]
        nx.draw_networkx_nodes(G, pos, node_color=label_colors)
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edges(G, pos, width=1.)
        plt.savefig('toy_edge_index.png')

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class Planetoid(InMemoryDataset):
    url = 'https://github.com/kimiyoung/planetoid/raw/master/data'

    def __init__(self, root, name, split="public", num_train_per_class=20,
                 num_val=500, num_test=1000, transform=None,
                 pre_transform=None):
        self.name = name

        super(Planetoid, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

        self.split = split
        assert self.split in ['public', 'full', 'full_60per', 'full_100per', 'random']

        if split == 'full':
            data = self.get(0)
            data.train_mask.fill_(True)
            data.train_mask[data.val_mask | data.test_mask] = False
            self.data, self.slices = self.collate([data])

        elif split == 'full_60per':
            data = self.get(0)
            idx_test = np.random.choice(2708, int(2708*0.2), replace=False)
            without_test=np.array([i for i in range(2708) if i not in idx_test])
            idx_train = without_test[np.random.choice(np.arange(len(without_test)),int(2708*0.6), replace=False)]
            idx_val = np.array([i for i in range(2708) if i not in idx_test if i not in idx_train])
            data.train_mask.fill_(False)
            data.val_mask.fill_(False)
            data.test_mask.fill_(False)

            for i in idx_train:
                data.train_mask[i] = True
            for i in idx_val:
                data.val_mask[i] = True
            for i in idx_test:
                data.test_mask[i] = True
            self.data, self.slices = self.collate([data])

        elif split == 'full_100per':
            data = self.get(0)
            data.train_mask.fill_(True)
            data.val_mask.fill_(True)
            data.test_mask.fill_(True)
            self.data, self.slices = self.collate([data])

        elif split == 'random':
            data = self.get(0)
            data.train_mask.fill_(False)
            for c in range(self.num_classes):
                idx = (data.y == c).nonzero(as_tuple=False).view(-1)
                idx = idx[torch.randperm(idx.size(0))[:num_train_per_class]]
                data.train_mask[idx] = True

            remaining = (~data.train_mask).nonzero(as_tuple=False).view(-1)
            remaining = remaining[torch.randperm(remaining.size(0))]

            data.val_mask.fill_(False)
            data.val_mask[remaining[:num_val]] = True

            data.test_mask.fill_(False)
            data.test_mask[remaining[num_val:num_val + num_test]] = True

            self.data, self.slices = self.collate([data])

    @property
    def raw_dir(self):
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed')

    @property
    def raw_file_names(self):
        names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
        return ['ind.{}.{}'.format(self.name.lower(), name) for name in names]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for name in self.raw_file_names:
            download_url('{}/{}'.format(self.url, name), self.raw_dir)

    def process(self):
        data = read_planetoid_data(self.raw_dir, self.name) # this is usual data i often use
        data = data if self.pre_transform is None else self.pre_transform(data) # transform __call__ is called
        torch.save(self.collate([data]), self.processed_paths[0]) # processed_paths[0] is 

    def __repr__(self):
        return '{}()'.format(self.name)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = FiveUniqueNodes(root='../data/toy', idx_train=[1,3,10,11])
    data = dataset[0].to(device)

    G = to_networkx(data)
    color_list = ['salmon' if data.y[idx]==0 else 'skyblue'
                  for idx in range(G.number_of_nodes())]
    colors = {idx: {'color': color_list[idx]} for idx in range(G.number_of_nodes())}
    nx.set_node_attributes(G, colors)

    pos = nx.spring_layout(G, k=0.7)
    label_colors = [node['color'] for node in G.nodes.values()]
    nx.draw_networkx_nodes(G, pos, node_color=label_colors)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, width=1.)
    plt.savefig('five_unique_nodes.png')