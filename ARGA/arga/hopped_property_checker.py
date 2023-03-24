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



data, hop = ['cora'], [0, 1]
print('Starting to calculate adjacency analysis for ' + data[0] + ' in hop ' + str(hop))

for i in range(len(data)):

    for j in range(len(hop)):
        a1 = 'pickles/' + data[i] + '_adj_hop_' + str(hop[j]) + '.pickle'
        with open(a1, 'rb') as handle: adj = pickle.load(handle)
        adj = adj.toarray()

        from collections import Counter
        res = dict(sum(map(Counter, adj), Counter()))
        print(res[1], res[0], res[1]+res[0])