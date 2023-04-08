import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pickle
import math
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from collections import defaultdict

from input_data import load_data
import scipy.sparse as sp

from hopInfo import addHopFeatures, addHopAdjacency

def store(features, adj, hop_count):
    # Manually store hopped info in pickle
    if hop_count != 0: 
        features = addHopFeatures(features, adj, hop_count)
        adj = sp.csr_matrix(adj)
        adj = addHopAdjacency(adj, hop_count + 1)

    f1 = 'pickles/' + data_name + '_features_hop_' + str(hop_count) + '.pickle'
    a1 = 'pickles/' + data_name + '_adj_hop_' + str(hop_count) + '.pickle'
    print('Stored in ', f1, 'and ', a1)
    with open(f1, 'wb') as handle: pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(a1, 'wb') as handle: pickle.dump(adj, handle, protocol=pickle.HIGHEST_PROTOCOL)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
import argparse
import sys
parser = argparse.ArgumentParser()

parser.add_argument('--dataname')
parser.add_argument('--hop')

# fix conflict with tf.app.flags with addition -- separator
# python development.py -- --dataname=cora --hop=1
args = parser.parse_args(sys.argv[2:])
print('Arguments:', args)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------


# Load data
data_name = args.dataname    # 'cora'
hop_count = int(args.hop)    # 2
print('Starting to calculate hopped feature and adjacency for' + data_name + ' with ' + str(hop_count))


adj, features, y_test, tx, ty, test_maks, true_labels = load_data(data_name)
store(features, adj, hop_count)
print('Done calculating hopped feature and adjacency for' + data_name + ' with ' + str(hop_count))
