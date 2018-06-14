import numpy as np
from collections import OrderedDict
import scipy.io as sio
import gensim
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer


class NodeClassification:
    """
    Multi-label node classification
    """

    def __init__(self, label_file, label_file_type, embedding_file, params):
        self.number_of_nodes = -1
        self.number_of_classes = -1
        self.label_matrix = []
        self.embeddings = []
        self.results = {}

        self.read_labels(label_file=label_file, file_type=label_file_type, params=params)
        self.read_embeddings(embedding_file=embedding_file)

    def read_labels(self, label_file, file_type, params={}):

        label_matrix_name = params['label_matrix_name']

        # Read the node labels
        if file_type == "mat":
            mat = sio.loadmat(label_file)
            self.label_matrix = mat[label_matrix_name]

        # Get the number of classes
        self.number_of_classes = self.label_matrix.shape[1]

    def read_embeddings(self, embedding_file):
        # Read the the embeddings
        model = gensim.models.KeyedVectors.load_word2vec_format(embedding_file, binary=False)

        # Get the number of nodes
        self.number_of_nodes = len(model.vocab)

        # Generate the embedding matrix
        self.embeddings = np.asarray([model[str(node)] for node in range(self.number_of_nodes)])

    def evaluate(self, number_of_shuffles, training_ratios):

        results = {}
        averages = ['micro', 'macro']
        for average in averages:
            results[average] = OrderedDict()
            for ratio in training_ratios:
                results[average].update({ratio: []})

        for train_ratio in training_ratios:

            for _ in range(number_of_shuffles):
                # Shuffle the data
                shuffle_features, shuffle_labels = shuffle(self.embeddings, self.label_matrix)

                # Get the training size
                train_size = int(train_ratio * self.number_of_nodes)

                # Divide the data into the training and test sets
                train_features = shuffle_features[0:train_size, :]
                train_labels = shuffle_labels[0:train_size]

                test_features = shuffle_features[train_size:, :]
                test_labels = shuffle_labels[train_size:]

                # Train the classifier
                ovr = OneVsRestClassifier(LogisticRegression())
                ovr.fit(train_features, train_labels)

                # Find the predictions, each node can have multiple labels
                test_prob = np.asarray(ovr.predict_proba(test_features))
                y_pred = []
                for i in range(test_labels.shape[0]):
                    k = test_labels[i].getnnz()  # The number of labels to be predicted
                    pred = test_prob[i, :].argsort()[-k:]
                    y_pred.append(pred)

                # Find the true labels
                y_true = [[] for _ in range(test_labels.shape[0])]
                co = test_labels.tocoo()
                for i, j in zip(co.row, co.col):
                    y_true[i].append(j)

                mlb = MultiLabelBinarizer(range(self.number_of_classes))
                for average in averages:
                    score = f1_score(y_true=mlb.fit_transform(y_true),
                                     y_pred=mlb.fit_transform(y_pred),
                                     average=average)

                    results[average][train_ratio].append(score)

        self.results = results

    def get_results(self, detailed=False):
        output = ""
        if detailed is True:
            for average in ["micro", "macro"]:
                output += average + "\n"
                for ratio in self.results[average]:
                    output += " percent {}%\n".format(ratio)
                    scores = self.results[average][ratio]
                    for num in range(len(scores)):
                        output += "  Shuffle #{}: {}\n".format(num, self.results[average][ratio][num])
                    output += "  Average: {} Std-dev: {}\n".format(np.mean(scores), np.std(scores))
                output += "\n"
        else:
            output += "Training percents: " + " ".join("{}%".format(p * 100) for p in self.results["micro"]) + "\n"
            for average in ["micro", "macro"]:
                output += average + ": "
                for ratio in self.results[average]:
                    scores = self.results[average][ratio]
                    output += "{0:.5g}:{1:.5g} ".format(np.mean(scores), np.std(scores))
                output += "\n"

        return output

    def print_results(self, detailed=False):
        output = self.get_results(detailed=detailed)
        print(output)

    def save_results(self, output_file, detailed=False):
        output = self.get_results(detailed=detailed)
        with open(output_file, 'w') as f:
            f.write(output)



