import json
import numpy as np
from six.moves import urllib
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
from networkx import from_numpy_array, from_numpy_matrix
from torch_geometric.nn.models import Node2Vec
import torch
import networkx as nx
import os
import sys

class RedditLoader(object):

    def __init__(self, dataset):
        self._read_local_data(dataset=dataset)

    def _read_local_data(self, dataset):
        dataset = dataset
        url = '../../dataset/'+str(dataset)+'.npz'
        self._dataset = np.load(url)
        self.Labels = self._dataset['labels']  # (n_node, num_classes)
        self.Graphs = self._dataset['adjs']  # (n_time, n_node, n_node)
        self.Features = self._dataset['attmats']  # (n_node, n_time, att_dim)

    def _get_edges(self):
        self._edges = []
        for time in range(len(self.Graphs)):
            graph = from_numpy_array(self.Graphs[time])
            #print('1111111111')
            #print(graph.get_edge_data(0,9)['weight'])
            #print(graph[0][9]['weight'])
            adj = nx.to_scipy_sparse_matrix(graph).tocoo()
            row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
            col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
            edge_index = torch.stack([row, col], dim=0)
            #print(edge_index)
            #exit()
            self._edges.append(edge_index.numpy())

    def _get_edge_weights(self):
        self._edge_weights = []
        for time in range(len(self.Graphs)):
            edges = self._edges[time].T
            #print(self.Graphs[time])
            graph = from_numpy_array(self.Graphs[time])
            #print(graph)
            #exit()
            edge_weight = []
            for e in edges:
                edge_weight.append(graph.get_edge_data(e[0],e[1])['weight'])
            self._edge_weights.append(np.array(edge_weight))

    def _get_targets_and_features(self):
        #print(torch.tensor(self.Features).shape)
        features = torch.tensor(self.Features).transpose(0, 1).numpy()
        self.features = [i for i in features]
        self.targets = [self.Labels for i in range(len(self.Graphs))]
        '''
        self.features = []
        url = 'node_features2.feature'
        if os.path.exists(url) == False:
            file = open(url, 'w')
        if os.path.getsize(url) > 0:
            self.features = torch.load(f=url)
        else:
            for time in range(len(self.Graphs)):
                feature = Node2Vec(edge_index=torch.tensor(self._edges[time]), num_nodes=self.Graphs[0].shape[0],
                                   embedding_dim=32, walk_length=8,walks_per_node=4,
                                   context_size=4).forward()
                self.features.append(np.array(feature.detach()))
            torch.save(self.features, f=url)
        '''
    def get_dataset(self) -> DynamicGraphTemporalSignal:
        """Returning the England COVID19 data iterator.

        Args types:
            * **lags** *(int)* - The number of time lags.
        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The England Covid dataset.
        """
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features()
        dataset = DynamicGraphTemporalSignal(self._edges, self._edge_weights, self.features, self.targets)
        return dataset