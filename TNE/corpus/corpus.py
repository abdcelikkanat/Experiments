import random
import graph as deepwalk
import node2vec
import networkx as nx


class Corpus:

    def __init__(self, nxg):
        self.graph = nxg
        self.number_of_nodes = nx.number_of_nodes(nxg)
        self.corpus = ""
        self.params = {}

    def graph2walks(self, method="", params={}):

        self.params = params

        if method == "deepwalk":
            number_of_walks = self.params['number_of_walks']
            walk_length = self.params['walk_length']
            alpha = self.params['alpha']

            # Temporarily generate the edge list
            with open("./temp/graph.edgelist", 'w') as f:
                for line in nx.generate_edgelist(self.graph, data=False):
                    f.write("{}\n".format(line))

            dwg = deepwalk.load_edgelist("./temp/graph.edgelist", undirected=True)
            corpus = deepwalk.build_deepwalk_corpus(G=dwg, num_paths=number_of_walks,
                                                    path_length=walk_length,
                                                    alpha=alpha,
                                                    rand=random.Random(0))

        elif method == "node2vec":

            number_of_walks = self.params['number_of_walks']
            walk_length = self.params['walk_length']
            p = self.params['p']
            q = self.params['q']

            for edge in self.graph.edges():
                self.graph[edge[0]][edge[1]]['weight'] = 1
            G = node2vec.Graph(nx_G=self.graph, p=p, q=q, is_directed=False)
            G.preprocess_transition_probs()
            corpus = G.simulate_walks(num_walks=number_of_walks, walk_length=walk_length)

        else:
            raise ValueError("Invalid method name!")

        """
        new_corpus = []
        line_counter = 0
        line = []
        for walk in corpus:
            if line_counter < self.params['number_of_walks']:
                line.extend(walk)
                line_counter += 1
            else:
                line_counter = 0
                new_corpus.append(line)
                line = []

        corpus = new_corpus
        """
        self.corpus = corpus

        return self.corpus

    def save(self, filename, with_title=False, save_one_line=False):

        with open(filename, 'w') as f:

            if save_one_line is True:

                if with_title is True:
                    f.write(u"{}\n".format(self.number_of_nodes))

                line_counter = 0
                line = []
                for walk in self.corpus:
                    if line_counter < self.params['number_of_walks']:
                        line.extend(walk)
                        line_counter += 1
                    else:
                        line_counter = 0
                        f.write(u"{}\n".format(u" ".join(v for v in line)))
                        line = []
            else:

                if with_title is True:
                    f.write(u"{}\n".format(self.number_of_nodes * self.params['number_of_walks']))

                for walk in self.corpus:
                    f.write(u"{}\n".format(u" ".join(v for v in walk)))


