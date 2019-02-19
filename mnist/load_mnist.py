from __future__ import print_function
import sys
try:
    import cPickle as pickle
except:
    import pickle
import datetime, math, sys, time

#from sklearn.datasets import fetch_mldata
import numpy as np

from chainer import cuda
from keras.datasets import mnist


class Data:
    def __init__(self, data, label):
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
    mnist = fetch_mldata('MNIST original', data_home=PATH)
    mnist.data = mnist.data.astype(np.float32)*scale + shift
    mnist.target = mnist.target.astype(np.int32)
    whole = Data(mnist.data, mnist.target)
    #(x_train, y_train), (x_test, y_test) = mnist.load_data()
    #x = np.concatenate((x_train, x_test))
    #y = np.concatenate((y_train, y_test))
    #x = x.astype(np.float32) * scale + shift
    #y = y.astype(np.int32)
    #whole = Data(x, y)

    print ("load mnist done", whole.data.shape)
    return whole

