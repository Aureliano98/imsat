from __future__ import print_function
#from .. import data
from data import Data 
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


def load_mnist_whole(scale, shift, PATH = '.'):
    print ('fetch MNIST dataset')
    mnist = scio.loadmat(os.path.join(PATH, 'mnist-original.mat'))
    x = np.transpose(mnist['data']).astype(np.float32) * scale + shift
    y = mnist['label'].astype(np.int32).flatten()
    whole = Data(x, y)

    print ("load mnist done", whole.data.shape)
    return whole

