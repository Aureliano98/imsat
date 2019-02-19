from __future__ import print_function
import sys
try:
    import cPickle as pickle
except:
    import pickle
import datetime, math, sys, time
import numpy as np
from chainer import cuda
import os
import scipy.io as scio

class Data:
    def __init__(self, data, label):
        assert len(data) == len(label)
        self.data = data
        self.label = label
        self.index = np.arange(len(data))

    def get_index_data(self, index_list):
        return cuda.to_gpu(self.data[index_list])

    def get(self, n, need_index = False):
        ind = np.random.permutation(self.data.shape[0])
        if need_index:
            return cuda.to_gpu(self.data[ind[:n],:].astype(np.float32)), cuda.to_gpu(self.label[ind[:n]].astype(np.int32)), self.index[ind[:n]].astype(np.int32)
        else:
            return cuda.to_gpu(self.data[ind[:n],:].astype(np.float32)), cuda.to_gpu(self.label[ind[:n]].astype(np.int32))

def load_mnist_whole(scale, shift, PATH = '.'):
    print ('fetch MNIST dataset')
    mnist = scio.loadmat(os.path.join(PATH, 'mnist-original.mat'))
    x = np.transpose(mnist['data']).astype(np.float32) * scale + shift
    y = mnist['label'].astype(np.int32).flatten()
    whole = Data(x, y)

    print ("load mnist done", whole.data.shape)
    return whole

