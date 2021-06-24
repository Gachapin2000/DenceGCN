import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import KarateClub

dataset = KarateClub(transform=None)
data = dataset[0]
max_depth = 4
G = to_networkx(data)
colors = ['silver', 'red', 'green', 'blue']
node_color = [colors[y.item()] for y in data.y]
nx.draw_networkx(G, node_color = node_color, arrows=False)
plt.savefig('./result/karate.png')