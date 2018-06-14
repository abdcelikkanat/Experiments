import sys
sys.path.append("../Evaluation/edge_prediction/")
sys.path.append("../TNE/")

from tne.tne import *

graph_path = "../Evaluation/examples/inputs/astro-ph.gml"
tne = TNE(graph_path)