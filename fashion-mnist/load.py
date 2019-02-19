from __future__ import print_function
from data import Data 
import numpy as np
import gzip


def _load_fashion_mnist(path) :
    """Loads the Fashion-MNIST dataset.
    Copied from keras.datasets.fashion_mnist.py

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    files = ['train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
             't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz']
    paths = [os.path.join(path, f) for f in files]

    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(imgpath.read(), np.uint8,
                                offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(imgpath.read(), np.uint8,
                               offset=16).reshape(len(y_test), 28, 28)

    return (x_train, y_train), (x_test, y_test)


def load_mnist_whole(scale, shift, path='.'):
    print ('fetch Fashion-MNIST dataset')
    (x_train, y_train), (x_test, y_test) = _load_fashion_mnist(path)
    # Respectively: (60k, 28, 28), (60k,), (10k, 28, 28), (10k,)

    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = np.transpose(x).astype(np.float32) * scale + shift
    y = y.astype(np.int32).flatten()
    whole = Data(x, y)

    print ("load Fashion-MNIST done", whole.data.shape)
    return whole


