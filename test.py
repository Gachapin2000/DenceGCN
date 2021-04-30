import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# 12個の頂点と、ランダムに引いた辺を持つグラフを定義
node_labels = "abcdefghijkl"
G = nx.Graph()
G.add_nodes_from(node_labels)
for i in range(len(G.nodes)):
    for j in range(i+1, len(G.nodes)):
        if np.random.uniform() < 0.3:
            G.add_edge(node_labels[i], node_labels[j])

pos = {'a': (1.0, 0.0),
       'b': (0.8660254037844387, 0.49999999999999994),
       'c': (0.5000000000000001, 0.8660254037844386),
       'd': (6.123233995736766e-17, 1.0),
       'e': (-0.4999999999999998, 0.8660254037844388),
       'f': (-0.8660254037844387, 0.49999999999999994),
       'g': (-1.0, 1.2246467991473532e-16),
       'h': (-0.8660254037844388, -0.4999999999999998),
       'i': (-0.5000000000000004, -0.8660254037844384),
       'j': (-1.8369701987210297e-16, -1.0),
       'k': (0.5, -0.8660254037844386),
       'l': (0.8660254037844384, -0.5000000000000004)
      }

# 座標を指定せずに描写する
nx.draw_networkx(G, node_color="c")
plt.show()