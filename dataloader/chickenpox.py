import io
import json
import numpy as np
from six.moves import urllib
from torch_geometric_temporal.signal import StaticGraphTemporalSignal

class ChickenpoxDatasetLoader(object):
    """A dataset of county level chicken pox cases in Hungary between 2004
    and 2014. We made it public during the development of PyTorch Geometric
    Temporal. The underlying graph is static - vertices are counties and 
    edges are neighbourhoods. Vertex features are lagged weekly counts of the 
    chickenpox cases (we included 4 lags). The target is the weekly number of 
    cases for the upcoming week (signed integers). Our dataset consist of more
    than 500 snapshots (weeks). 
    """
    def __init__(self):
        #self._read_web_data()
        self._read_local_data()


    def _read_web_data(self):
        url = "https://raw.githubusercontent.com/benedekrozemberczki/pytorch_geometric_temporal/master/dataset/chickenpox.json"
        #url = ""
        self._dataset = json.loads(urllib.request.urlopen(url).read())

    def _read_local_data(self):
        #url = "D:/PycharmProjects/pytorch_geometric_temporal-master (2)/pytorch_geometric_temporal-master/dataset/chickenpox.json"
        url = "../../dataset/chickenpox.json"
        with open(url, 'r') as f:
            self._dataset = json.load(f)



    def _get_edges(self):
        self._edges = np.array(self._dataset["edges"]).T
        #102条边
        '''
        print('11111111111')
        print(self._edges)
        '''
        #print(len(self._edges[0]))


    def _get_edge_weights(self):
        self._edge_weights = np.ones(self._edges.shape[1])
        #权重102个1
        '''
        print('2222222222')
        #print(self._edge_weights)
        print(len(self._edge_weights))
        print(self._edge_weights)
        '''

    def _get_targets_and_features(self):
        stacked_target = np.array(self._dataset["FX"])
        self.features = [stacked_target[i:i+self.lags,:].T for i in range(stacked_target.shape[0]-self.lags)]
        self.targets = [stacked_target[i+self.lags,:].T for i in range(stacked_target.shape[0]-self.lags)]
        '''
        print('33333')
        print(len(self.targets))
        print(len(self.targets[0]))
        print(self.targets[0])
        #print(len(self.targets[0][0]))
        print(len(self.features[0][0]))
        print(len(self.features[0]))
        print(len(self.features))
        #print(self.targets)
        print('444')
        '''
        '''
        33333
        489
        20
        32
        20
        489
        20
        '''

    def get_dataset(self, lags: int=4) -> StaticGraphTemporalSignal:
        """Returning the Chickenpox Hungary data iterator.

        Args types:
            * **lags** *(int)* - The number of time lags. 
        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The Chickenpox Hungary dataset.
        """
        self.lags = lags
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features()
        dataset = StaticGraphTemporalSignal(self._edges, self._edge_weights, self.features, self.targets)
        return dataset
