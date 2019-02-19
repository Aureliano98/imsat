from __future__ import print_function
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


def load_whole(scale, shift, path='.'):
    print ('fetch MNIST dataset')
    mnist = scio.loadmat(os.path.join(path, 'mnist-original.mat'))
    x = np.transpose(mnist['data']).astype(np.float32) * scale + shift
    y = mnist['label'].astype(np.int32).flatten()
    whole = Data(x, y)

    print ("load mnist done", whole.data.shape)
    return whole

