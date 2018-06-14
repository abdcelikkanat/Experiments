from os.path import basename, splitext
from tne.tne import TNE
from utils.utils import *
import matplotlib.pylab as plt
import networkx as nx
import time

dataset_folder = "../datasets/"
outputs_folder = "../outputs/"
temp_folder = "../temp/"

dataset_file = "karate.gml"

method = "deepwalk"

number_of_topics = 3
number_of_iters = 2000
params = {}
params['number_of_walks'] = 80
params['walk_length'] = 40

params['window_size'] = 10
params['embedding_size'] = 128
params['alpha'] = 0
params['p'] = 1.0
params['q'] = 1.0
alpha = 50.0/float(number_of_topics)
beta = 0.1


base_desc = "figure_{}_n{}_l{}_w{}_k{}_{}.corpus".format(splitext(basename(dataset_file))[0],
                                                            params['number_of_walks'],
                                                            params['walk_length'],
                                                            params['window_size'],
                                                            number_of_topics,
                                                            method)

graph_path = dataset_folder + dataset_file
tne = TNE(graph_path)
tne.perform_random_walks(method="deepwalk", params=params)


corpus_path_for_lda = temp_folder + "{}.corpus".format(base_desc)

tne.save_corpus(corpus_path_for_lda, with_title=True)
id2node = tne.run_lda(alpha=alpha, beta=beta, number_of_iters=number_of_iters,
                      number_of_topics=number_of_topics, lda_corpus_path=corpus_path_for_lda)

phi_file = tne.get_file_path(filename="phi")
node2topic = find_max_topic_for_nodes(phi_file, id2node, number_of_topics)




plt.figure()
g = tne.get_nxgraph()
pos = nx.spring_layout(g)
color = ['r', 'b', 'g', 'y', 'm', 'c']
# nodes
for t in range(number_of_topics):
    nx.draw_networkx_nodes(g, pos, nodelist=[node for node in g.nodes() if node2topic[str(node)]==t],
                           node_color="tbl:red",
                           node_size=200,
                           alpha=0.6,
                           linewidths=2)
nx.draw_networkx_edges(g, pos,
                       edgelist=g.edges(),
                       width=1, alpha=0.5)
#plt.show()

plt.axis('off')
plt.savefig("../datasets/graph3.png", format="eps",  bbox_inches="tight")