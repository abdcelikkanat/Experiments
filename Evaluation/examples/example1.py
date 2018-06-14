from classification.multi_label_node import *

label_file = "./inputs/citeseer.mat"
embedding_file = "./inputs/citeseer_n40_l10_w10_deepwalk_node.embedding"
label_file_type = "mat"
params = {"label_matrix_name": "group"}

nc = NodeClassification(label_file, label_file_type, embedding_file, params)
number_of_shuffles = 50
training_ratios = np.arange(1, 10)*0.1
nc.evaluate(number_of_shuffles=number_of_shuffles,
            training_ratios=training_ratios)
nc.save_results("./outputs/test.result", detailed=False)