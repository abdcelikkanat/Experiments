import os
import sys
import time
import random
import networkx as nx
from utils.utils import *
from ext.gensim_wrapper.models.word2vec import Word2VecWrapper, CombineSentences, LineSentence
from gensim.utils import smart_open

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../ext/deepwalk/deepwalk")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../ext/node2vec/src")))
lda_exe_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../ext/gibbslda/lda"))


try:
    import graph as deepwalk
    import node2vec
    if not os.path.exists(lda_exe_path):
        raise ImportError
except ImportError:
    raise ImportError("An error occurred during loading the external libraries!")


class WalkIterator:
    def __init__(self, corpus):
        self.corpus = corpus

    def __iter__(self):
        for walk in self.corpus:
            yield [str(w) for w in walk]


class TNE:
    def __init__(self, graph_path=None):
        self.graph = None
        self.graph_name = ""
        self.number_of_nodes = 0
        self.method = ""
        self.corpus = []
        self.params = {}
        self.model = None

        self.temp_folder = "../temp/"
        self.node_corpus_path = ""
        self.topic_corpus_path = ""

        self.lda_corpus_dir = ""
        self.lda_wordmapfile = ""
        self.lda_tassignfile = ""
        self.lda_node_corpus = ""
        self.lda_topic_corpus = ""
        self.lda_phi_file = ""
        self.lda_theta_file = ""

        if graph_path is not None:
            self.read_graph(graph_path)

        if not os.path.exists(self.temp_folder):
            os.makedirs(self.temp_folder)

    def read_graph(self, filename, filetype=".gml"):
        dataset_name = os.path.splitext(os.path.basename(filename))[0]

        if filetype == ".gml":
            g = nx.read_gml(filename)
            print("Dataset: {}".format(dataset_name))
            print("The number of nodes: {}".format(g.number_of_nodes()))
            print("The number of edges: {}".format(g.number_of_edges()))

            self.number_of_nodes = g.number_of_nodes()
            self.graph = g
            self.graph_name = dataset_name
        else:
            raise ValueError("Invalid file type!")

    def set_graph(self, graph, graph_name="unknown"):

        self.graph = graph
        self.number_of_nodes = graph.number_of_nodes()
        self.graph_name = graph_name

        print("Graph name: {}".format(self.graph_name))
        print("The number of nodes: {}".format(self.graph.number_of_nodes()))
        print("The number of edges: {}".format(self.graph.number_of_edges()))

    def perform_random_walks(self, method, params):

        initial_time = time.time()
        # Generate a corpus

        if method == "deepwalk":
            if not ('number_of_walks' and 'walk_length' and 'alpha') in params.keys():
                raise ValueError("A missing parameter exists!")

            self.params = params

            # Temporarily generate the edge list
            with smart_open(self.temp_folder + "graph_deepwalk.edgelist", 'w') as f:
                for line in nx.generate_edgelist(self.graph, data=False):
                    f.write("{}\n".format(line))

            dwg = deepwalk.load_edgelist(self.temp_folder + "graph_deepwalk.edgelist", undirected=True)
            self.corpus = deepwalk.build_deepwalk_corpus(G=dwg, num_paths=self.params['number_of_walks'],
                                                         path_length=self.params['walk_length'],
                                                         alpha=self.params['alpha'],
                                                         rand=random.Random(0))

        elif method == "node2vec":

            if not ('number_of_walks' and 'walk_length' and 'p' and 'q') in params.keys():
                raise ValueError("A missing parameter exists!")

            self.params = params

            for edge in self.graph.edges():
                self.graph[edge[0]][edge[1]]['weight'] = 1
            G = node2vec.Graph(nx_G=self.graph, p=self.params['p'], q=self.params['q'], is_directed=False)
            G.preprocess_transition_probs()
            self.corpus = G.simulate_walks(num_walks=self.params['number_of_walks'],
                                           walk_length=self.params['walk_length'])

        else:
            raise ValueError("Invalid method name!")

        self.node_corpus_path = os.path.join(self.temp_folder, "node.corpus")
        self.save_corpus(self.node_corpus_path, with_title=False)

        self.method = method
        self.params = params

        print("The corpus was generated in {:.2f} secs.".format(time.time() - initial_time))

    def save_corpus(self, corpus_file, with_title=False, corpus=None):

        # Save the corpus
        with smart_open(corpus_file, "w") as f:

            if with_title is True:
                f.write(u"{}\n".format(self.number_of_nodes * self.params['number_of_walks']))

            if corpus is None:
                for walk in self.corpus:
                    f.write(u"{}\n".format(u" ".join(v for v in walk)))
            else:
                for walk in corpus:
                    f.write(u"{}\n".format(u" ".join(v for v in walk)))

    def extract_node_embedding(self, node_embedding_file, workers=3):

        initial_time = time.time()

        # Extract the node embeddings
        self.model = Word2VecWrapper(sentences=LineSentence(self.node_corpus_path),
                                     size=self.params["embedding_size"],
                                     window=self.params["window_size"],
                                     sg=1, hs=1,
                                     workers=workers,
                                     min_count=0)

        # Save the node embeddings
        self.model.wv.save_word2vec_format(fname=node_embedding_file)
        print("The node embeddings were generated and saved in {:.2f} secs.".format(time.time() - initial_time))

    def run_lda(self, alpha, beta, number_of_iters, number_of_topics, lda_corpus_path):

        self.lda_corpus_dir = os.path.dirname(os.path.join(lda_corpus_path))
        self.lda_wordmapfile = os.path.join(self.lda_corpus_dir, "wordmap.txt")
        self.lda_tassignfile = os.path.join(self.lda_corpus_dir, "model-final.tassign")
        self.lda_node_corpus = os.path.join(self.lda_corpus_dir, "lda_node.file")
        self.lda_topic_corpus = os.path.join(self.lda_corpus_dir, "lda_topic.file")
        self.lda_phi_file = os.path.join(self.lda_corpus_dir, "model-final.phi")
        self.lda_theta_file = os.path.join(self.lda_corpus_dir, "model-final.theta")

        initial_time = time.time()
        # Run GibbsLDA++
        cmd = "{} -est ".format(lda_exe_path)
        cmd += "-alpha {} ".format(alpha)
        cmd += "-beta {} ".format(beta)
        cmd += "-ntopics {} ".format(number_of_topics)
        cmd += "-niters {} ".format(number_of_iters)
        cmd += "-savestep {} ".format(number_of_iters+1)
        cmd += "-dfile {} ".format(lda_corpus_path)
        os.system(cmd)

        # Generate the id2node dictionary
        id2node = generate_id2node(self.lda_wordmapfile)
        print("-> The LDA algorithm run in {:.2f} secs".format(time.time() - initial_time))

        assert len(id2node) == self.number_of_nodes, "LDA could not run well!"

        return id2node

    def get_topic_corpus(self):
        topic_corpus = []
        with smart_open(self.lda_tassignfile, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                topic_corpus.append([token.split(':')[1] for token in tokens])

        return topic_corpus

    def extract_topic_embedding(self, number_of_topics, topic_embedding_file):
        # Define the paths for the files generated by GibbsLDA++
        initial_time = time.time()
        # Convert node corpus to the corresponding topic corpus
        topic_corpus = self.get_topic_corpus()
        self.topic_corpus_path = os.path.join(self.temp_folder, "topic.corpus")
        self.save_corpus(corpus_file=self.topic_corpus_path,  with_title=False, corpus=topic_corpus)

        # Construct the tuples (word, topic) with each word in the corpus and its corresponding topic assignment
        combined_sentences = CombineSentences(self.node_corpus_path, self.topic_corpus_path)
        # Extract the topic embeddings
        self.model.train_topic(number_of_topics, combined_sentences)
        # Save the topic embeddings
        self.model.wv.save_word2vec_topic_format(fname=topic_embedding_file)
        print("The topic embeddings were generated and saved in {:.2f} secs.".format(time.time() - initial_time))

    def get_file_path(self, filename):

        if filename == "phi":
            return os.path.realpath(self.lda_phi_file)

        if filename == "theta":
            return os.path.realpath(self.lda_theta_file)

    def get_nxgraph(self):
        return self.graph

    def get_lda_corpus_path(self):
        return self.lda_node_corpus

    def get_node_corpus_path(self):
        return self.node_corpus_path

    def get_topic_corpus_path(self):
        return self.topic_corpus_path