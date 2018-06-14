import networkx as nx


path = "./p2p-Gnutella08.edgelist"
output = "./p2p-Gnutella08.gml"

g = nx.Graph()

with open(path, 'r') as f:
	for line in f:
		tokens = line.strip().split()
		for t in tokens[1:]:
			g.add_edge(tokens[0], t)

print("n: {}, e: {}".format(g.number_of_nodes(), g.number_of_edges()))

nx.write_gml(g, output)
