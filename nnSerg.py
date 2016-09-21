import numpy as np
import math
from sklearn.preprocessing import LabelBinarizer
from sklearn.datasets import make_moons

def _sigmoid(x): return 1 / (1 + math.exp(-x))


def _d_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)

def _tanH(x):
    return np.tanh(x)


sigmoid = np.vectorize(_sigmoid)
d_sigmoid = np.vectorize(_d_sigmoid)
tanH = np.vectorize(_tanH)

def activate(act):
    """
    Returns a function and derivative tuple
    :param act: 'sigmoid', 'tanh', or 'ReLU'
    :return: a tuple of functions (fun, grad)
    """
    if act == 'sigmoid': return (sigmoid, d_sigmoid)
    if act == 'tanh' : return (tanH)

def weight_matrix(innum, outnum):
    """
    Returns randomly initialized weight matrix of appropriate dimensions
    :param innum: number of neurons in the layer i
    :param outnum: number of neurons in layer i+1
    :return: weight matrix
    """
    W = np.random.rand(outnum, innum + 1)
    return W


def nn_build(layerlist, activation='tanH', eta=0.01):
    """
    Returns a neural network of given architecture
    :param layerlist: a list of number of neurons in each layer starting with input and ending with output
    :param activation: an activation function, which is either 'sigmoid', 'tanh', or 'ReLU'
    :return: a discionary with neural network parameters
    """
    nn = {}
    nn['eta'] = eta
    nn['layers'] = []
    nn['activations'] = []
    for i in range(len(layerlist) - 1):
        nn['layers'].append(weight_matrix(layerlist[i], layerlist[i + 1]))
        nn['activations'].append(activate(activation))
    return nn


def forward(nn, data):
    nn['inputs'] = [np.concatenate((data, [np.ones(data.shape[1])]))]
    r = np.concatenate((data, [np.ones(data.shape[1])]))
    for l, a in zip(nn['layers'], nn['activations']):
        r = a[0](np.dot(l, r))
        r = np.concatenate((r, [np.ones(r.shape[1])]))
        nn['inputs'].append(r)
    r = np.delete(r, (r.shape[0] - 1), axis=0)
    return r


def backprop(nn, err):
    eta = nn['eta']
    deltas = [err]
    for i in range(len(nn['layers'])-1, -1, -1):
        deltas.append(np.dot(nn['layers'][i].T, deltas[-1][:-1,:]))
    nn['deltas'] = deltas[::-1]

    for i in range(len(nn['layers'])):
        dact = nn['activations'][i][1]
        dW = 0
        ds = dact(nn['deltas'][i+1][:-1,:]) * nn['deltas'][i+1][:-1,:]
        samples = nn['inputs'][i].shape[1]
        for j in range(samples):
            dW = dW + np.outer(ds[:,j],nn['inputs'][i][:,j])
        nn['layers'][i] += eta * dW/samples


def expand_labels(labels):
    n = len(np.unique(labels))
    #if n == 2:
    #    l = np.array([(0,1) if row==0 else (1,0)  for row in labels]).T
    #else:
    lb = LabelBinarizer()
    l = lb.fit_transform(labels).T
    return l


def minibatch_fit(nn, data, labels):
    r = forward(nn, data)
    delta = np.concatenate((labels - r,[np.ones(r.shape[1])]))
    backprop(nn, delta)
    #ipdb.set_trace()
    return np.sum(np.square(r - labels))/2

data, label = make_moons(n_samples=300, noise=0.4)
nn = nn_build(expand_labels(label))
minibatch_fit(nn,data,expand_labels(label))