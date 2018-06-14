import networkx as nx
import numpy as np
import scipy.io as sio
from scipy.sparse import csr_matrix



mat_file_path = "./citeseer.mat"
output_file = "./citeseer_undirected.gml"
directed = False

# Set mat file parameters
mat_network_name = "network"
mat_cluster_name = "group"


# Read the mat file
mat_dict = sio.loadmat(mat_file_path)

adj_matrix = csr_matrix(mat_dict[mat_network_name])
cluster_matrix = csr_matrix(mat_dict[mat_cluster_name])

N = adj_matrix.shape[1]
K = cluster_matrix.shape[1]
E = adj_matrix.count_nonzero()
# Print graphs statistics
print("Number of nodes: {}".format(N))
print("Number of edges: {}".format(E))
print("Number of clusters: {}".format(K))

# Convert it into a undirected graph
g = nx.DiGraph()

cx = adj_matrix.tocoo()
for i, j, val in zip(cx.row, cx.col, cx.data):
    if val > 0:
        g.add_edge(str(i), str(j))

assert g.number_of_nodes() == N, "There is an error, the number of nodes mismatched, {} != {}".format(N, g.number_of_nodes())
assert g.number_of_edges() == E, "There is an error, the number of edges mismatched, {} != {}".format(E, g.number_of_edges())

if directed is False:
    g = g.to_undirected()
    print("The graph was converted into undirected!")
    print("Number of nodes: {}".format(g.number_of_nodes()))
    print("Number of edges: {}".format(g.number_of_edges()))

# Read clusters
values = {node: [] for node in g.nodes()}

cx = cluster_matrix.tocoo()
for i, k, val in zip(cx.row, cx.col, cx.data):
    if val > 0:
        values[str(i)].append(str(k))

nx.set_node_attributes(g, name="clusters", values=values)

# Finally, save the file
nx.write_gml(g, output_file)
