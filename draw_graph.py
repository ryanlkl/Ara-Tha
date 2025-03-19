import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

adj_matrix = np.array(
    [[0, 0, 0, 0, 0, 0, 0, -2, -1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
    [-2, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0, -3],
    [0, 0, 0, -2, -1, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0],
    [0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0,-3, 0, 0, 0, 0,-1, 0],
    [0, 0, 0, 0, 0, 0, 0,-2, 0, 0, 0, 0],
    [0, 0, 0,-2, 0,-2, 0, 0, 0, 0,-1, 0],
    [0, 0, 0, 0,-3, 0,-1, 0, 0,-3,-1, 0],
    [0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0]]
)

G = nx.from_numpy_array(adj_matrix)

nx.draw(G, with_labels=True, node_size=500, node_color="skyblue", font_size=10)
plt.show()