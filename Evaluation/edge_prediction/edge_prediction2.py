import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, model_selection, pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

class EdgePrediction:
    def __init__(self):
        self.g = None

    def read_graph(self, file_path, file_type="gml"):
        if file_type == "gml":
            self.g = nx.read_gml(file_path)
            self.number_of_nodes = self.g.number_of_nodes()
            self.number_of_edges = self.g.number_of_edges()
        else:
            raise ValueError("Unknown graph type!")

        print("Number of nodes: {}".format(self.number_of_nodes))
        print("Number of edges: {}".format(self.number_of_edges))

    def split_into_train_test_sets(self, ratio, max_trial_limit=10000):

        test_set_size = int(ratio * self.number_of_edges)
        train_set_size = self.number_of_edges - test_set_size

        # Generate the positive test edges
        test_pos_samples = []
        residual_g = self.g.copy()
        num_of_ccs = nx.number_connected_components(residual_g)
        if num_of_ccs != 1:
            raise ValueError("The graph contains more than one connected component!")

        num_of_pos_samples = 0

        edges = list(residual_g.edges())
        perm = np.arange(len(edges))
        np.random.shuffle(perm)
        edges = [edges[inx] for inx in perm]
        for i in range(len(edges)):

            # Remove the chosen edge
            chosen_edge = edges[i]
            residual_g.remove_edge(chosen_edge[0], chosen_edge[1])

            if chosen_edge[1] in nx.connected._plain_bfs(residual_g, chosen_edge[0]):
                num_of_pos_samples += 1
                test_pos_samples.append(chosen_edge)
                print("\r{0} tp edges found out of {1}".format(num_of_pos_samples, test_set_size)),
            else:
                residual_g.add_edge(chosen_edge[0], chosen_edge[1])

            if num_of_pos_samples == test_set_size:
                break

        if num_of_pos_samples != test_set_size:
            raise ValueError("Not pos edges found!")


        # Generate the negative samples
        test_neg_samples = []

        non_edges = list(nx.non_edges(self.g))
        perm = np.arange(len(non_edges))
        np.random.shuffle(perm)
        non_edges = [non_edges[inx] for inx in perm]

        chosen_non_edge_inx = np.random.choice(perm, size=test_set_size, replace=False)

        test_neg_samples = [non_edges[perm[p]] for p in chosen_non_edge_inx]

        """
        while num_of_removed_edges < test_set_size:
            # Randomly choose an edge index
            pos_inx = np.arange(residual_g.number_of_edges())
            np.random.shuffle(pos_inx)
            edge_inx = np.random.choice(a=pos_inx)
            # Remove the chosen edge
            chosen_edge = list(residual_g.edges())[edge_inx]
            residual_g.remove_edge(chosen_edge[0], chosen_edge[1])

            #reachable_from_v1 = nx.connected._plain_bfs(self.G, edge[0])
            if chosen_edge[1] in nx.connected._plain_bfs(residual_g, chosen_edge[0]):
                num_of_removed_edges += 1
                test_pos_samples.append(chosen_edge)
                trial_counter = 0
            else:
                residual_g.add_edge(chosen_edge[0], chosen_edge[1])
                trial_counter += 1

            if trial_counter == max_trial_limit:
                raise ValueError("In {} trial, any possible edge for removing could not be found!")

            print("\r{0} tp edges found out of {1}".format(num_of_removed_edges, test_set_size)),
        
        # Generate the negative samples
        test_neg_samples = []

        num_of_neg_samples = 0
        while num_of_neg_samples < test_set_size:

            pos_inx = np.arange(self.g.number_of_nodes())
            np.random.shuffle(pos_inx)
            # Self-loops are allowed
            u, v = np.random.choice(a=pos_inx, size=2)

            candiate_edge = (unicode(u), unicode(v))
            if not self.g.has_edge(candiate_edge[0], candiate_edge[1]) and candiate_edge not in self.g.edges():
                test_neg_samples.append(candiate_edge)
                num_of_neg_samples += 1

            print("\r{0} fn edges found out of {1}".format(num_of_neg_samples, test_set_size)),
        """

        return residual_g, test_pos_samples, test_neg_samples

    def train(self, train_graph, test_edges):

        pos_samples, neg_samples = test_edges
        n = train_graph.number_of_nodes()

        coeff_matrix = np.zeros(shape=(n, n), dtype=np.float)

        samples = pos_samples + neg_samples

        preds = nx.jaccard_coefficient(train_graph, samples)
        for i, j, p in preds:
            coeff_matrix[int(i), int(j)] = p
            coeff_matrix[int(j), int(i)] = p

        coeff_matrix = coeff_matrix / coeff_matrix.max()

        ytrue = [1 for _ in range(len(pos_samples))] + [0 for _ in range(len(neg_samples))]
        y_score = [coeff_matrix[int(edge[0]), int(edge[1])] for edge in pos_samples] + [coeff_matrix[int(edge[0]), int(edge[1])] for edge in neg_samples]

        auc = roc_auc_score(y_true=ytrue, y_score=y_score)

        print(auc)

        return auc

    def compute_features(self, nxg, edges, metric):

        features = []
        if metric == "jaccard":
            for i in range(len(edges)):
                for _, _, val in nx.jaccard_coefficient(nxg, [edges[i]]):
                   features.append(val)

            features = np.asarray(features)

        if metric == "node2vec":

            #features = (features - np.min(features))

            #features = features / np.max(features)
        return features

    def run(self, metric):
        train_residual_g, train_pos, train_neg = self.split_into_train_test_sets(ratio=0.7)
        #test_residual_g, test_pos, test_neg = self.split_into_train_test_sets(ratio=0.003)

        train_samples = train_pos + train_neg
        train_labels = [1 for _ in train_pos] + [0 for _ in train_neg]
        print(train_residual_g.number_of_edges())
        print(len(train_pos))
        print(len(train_neg))
        """
        train_features = self.compute_features(nxg=train_residual_g, edges=train_samples, metric=metric)

        test_samples = test_pos + test_neg
        test_labels = [1 for _ in test_pos] + [0 for _ in test_neg]
        test_features = self.compute_features(nxg=test_residual_g, edges=test_samples, metric=metric)

        train_score = roc_auc_score(y_true=train_labels, y_score=train_features)
        test_score = roc_auc_score(y_true=test_labels, y_score=test_features)


        print("Train: {}".format(train_score))
        print("Test: {}".format(test_score))

        scaler = StandardScaler()
        lin_clf = LogisticRegression()
        clf = pipeline.make_pipeline(scaler, lin_clf)

        train_features = [[f] for f in train_features]
        test_features = [[f] for f in test_features]

        train_features, train_labels = shuffle(train_features, train_labels)
        test_features, test_labels = shuffle(test_features, test_labels)

        # Train classifier
        clf.fit(train_features, train_labels)
        auc_train = metrics.scorer.roc_auc_scorer(clf, train_features, train_labels)
        avg = metrics.scorer.average_precision_scorer(clf, train_features, train_labels)
        # Test classifier
        auc_test = metrics.scorer.roc_auc_scorer(clf, test_features, test_labels)
        print("Train-Test: {}, {}, {}".format(auc_train, auc_test, avg))
        """

graph_path = "../examples/inputs/facebook.gml"
ep = EdgePrediction()
ep.read_graph(file_path=graph_path, file_type="gml")
ep.run(metric="jaccard")
#train, pos, neg = ep.split_into_train_test_sets(ratio=0.5)
#m = ep.train(ep.g, (pos, neg))
#print(m)

#https://github.com/adocherty/node2vec_linkprediction/blob/master/link_prediction.py
#node2vec link prediction github
