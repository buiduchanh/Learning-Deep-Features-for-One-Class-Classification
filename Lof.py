import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neighbors import KDTree
_contamination = 0.1
from math import sqrt
from queue import Queue
import csv

class Caculate(object):
    def __init__(self, data):
        self.tree = KDTree(data, leaf_size=30, metric='euclidean')
        self.dist = None
        self._lrd = None
    
    def get_query (self, data):

        dist_ = []
        index_ = []
        for idx, item in enumerate(data):
            last_kc = []
            last_index = []
            kc = []
            index = []
            for i, data_ in enumerate(data):
                if i != idx :
                    distance = [pow((data[i][j] - data[idx][j]), 2) for j in range (data[i].shape[0])]
                    dist = sqrt(sum(distance))
                    kc.append(dist)
                    index.append(i)

            kc = np.array(kc)
            index = np.array(index)

            xy = zip(kc, index)
            xy = sorted(xy, key = lambda x : x[0])[:5]
            for items in xy:
                print()
                last_kc.append(items[0])
                last_index.append(items[1])

            dist_.append(last_kc)
            index_.append(last_index)

        return np.array(dist_), np.array(index_)

    def compute(self, data, k , train = True):

        dist, ind = self.tree.query(data, k)
        # dist , ind = self.get_query(data)
        print('ind_train', ind.shape, np.max(ind))
        if train:
            dist = dist[:,1:]
            ind = ind[:,1:]
            self.dist = dist
            dist_k = self.dist[ind, k - 2]
            reach_dist_array = np.maximum(dist, dist_k)
            _lrd = 1. / (np.mean(reach_dist_array, axis=1) + 1e-10)
            self._lrd = _lrd
            lrd_ratios_array = (_lrd[ind] / _lrd[:, np.newaxis])
            
            print(np.mean(lrd_ratios_array, axis=1))
        else:
            print('ind_test',ind.shape, np.max(ind))
            dist_k = self.dist[ind, k - 1]
            reach_dist_array = np.maximum(dist, dist_k)
            _lrd = 1. / (np.mean(reach_dist_array, axis=1) + 1e-10)
            lrd_ratios_array = (self._lrd[ind] / _lrd[:, np.newaxis])
        
        # if not train:
            # print(dist[:1])
            # print(ind[:1])
            # print(_lrd[:1])
            # print(lrd_ratios_array[:1])
        return -np.mean(lrd_ratios_array, axis=1)

traindata = []
normal = []
abnormal = []

caculate = Caculate(X)

negative_outlier_factor_ = caculate.compute(X , 11, train= True)
exit()
offset_ = np.percentile(negative_outlier_factor_, 100. * _contamination)
result = caculate.compute(test, 5,train= False)
