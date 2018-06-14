import numpy as np
import networkx as nx
from scipy.io import loadmat
from scipy.sparse import issparse


def to_networkx(x, undirected=True):

	g = nx.Graph()

	if issparse(x):
		if undirected is True:
			cx = x.tocoo()
			for i, j, v in zip(cx.row, cx.col, cx.data):
				g.add_edge(str(i), str(j))
		else:
			raise Exception("Not implemented for directed case!")
	else:
		raise Exception("Not implemented for dense case")

	print("n: {}, e: {}".format(g.number_of_nodes(), g.number_of_edges()))
	return g


def load_matfile(filename, variable_name="network", undirected=True):
	print(filename)
	mat = loadmat(filename)
	#print(mat)
	mat_matrix = mat[variable_name]
	

	return to_networkx(mat_matrix, undirected)


path = "./dblp.mat"
output = "./dblp.gml"
g = load_matfile(path, variable_name="network", undirected=True)
nx.write_gml(g, output)
