import numpy as np
import math
from sklearn.preprocessing import LabelBinarizer
import pylab as plt
#import ipdb


def _relu(x, eps=1e-5): return max(eps, x)


def _d_relu(x, eps=1e-5): return 1. if x > eps else 0.0


def _sigmoid(x): return 1 / (1 + math.exp(-x))


def _tanh(x): return math.tanh(x)


def _d_tanh(x):
    t = _tanh(x)
    return 1 - t*t


def _d_sigmoid(x):
    s = _sigmoid(x)
    return s * (1 - s)

#2 sites, 50 items.  To 50 sites, 2 items
def d_cost(output, target): return output - target


sigmoid = np.vectorize(_sigmoid)
d_sigmoid = np.vectorize(_d_sigmoid)
relu = np.vectorize(_relu)
d_relu = np.vectorize(_d_relu)
tanh = np.vectorize(_tanh)
d_tanh = np.vectorize(_d_tanh)


def activate(act):
    """
    Returns a function and derivative tuple
    :param act: 'sigmoid', 'tanh', or 'ReLU'
    :return: a tuple of functions (fun, grad)
    """
    if act == 'sigmoid': return (sigmoid, d_sigmoid)
    if act == 'relu': return (relu, d_relu)
    if act == 'tanh': return (tanh, d_tanh)

def addadadelta(nn):
    # need to add error checking if weights have been already initialized
    nn['E2'] = {}
    nn['E2']['W'] = []
    nn['E2']['b'] = []
    nn['EW2'] = {}
    nn['EW2']['W'] = []
    nn['EW2']['b'] = []
    for i in range(len(nn['weights'])):
        nn['E2']['W'].append(np.zeros(nn['weights'][i].shape))
        nn['E2']['b'].append(np.zeros(nn['biases'][i].shape))
        nn['EW2']['W'].append(np.zeros(nn['weights'][i].shape))
        nn['EW2']['b'].append(np.zeros(nn['biases'][i].shape))
        
def weight_matrix(seed, innum, outnum, type='glorot', layer=0):
    """
    Returns randomly initialized weight matrix of appropriate dimensions
    :param innum: number of neurons in the layer i
    :param outnum: number of neurons in layer i+1
    :return: weight matrix
    """
    np.random.seed(seed)
    if type == 'glorot': W = np.random.uniform(low=-np.sqrt(6.0/(2*layer+1)), high=np.sqrt(6.0/(2*layer+1)), size=(outnum, innum))
    if type == 'normal': W = np.random.rand(outnum, innum)   
    return W


def nn_build(seed,layerlist, nonlin='sigmoid', eta=0.01, init='glorot'):
    """
    Returns a neural network of given architecture
    :param layerlist: a list of number of neurons in each layer starting with input and ending with output
    :param nonlin: nonlinearity which is either 'sigmoid', 'tanh', or 'ReLU'
    :return: a dictionary with neural network parameters
    """
    nn = {}
    nn['eta'] = eta
    nn['weights'] = []
    nn['biases'] = []
    nn['nonlin'] = []
    for i in range(len(layerlist) - 1):
        nn['weights'].append(weight_matrix(seed,layerlist[i], layerlist[i + 1],layer=i,type=init))
        nn['biases'].append(np.ones(layerlist[i + 1]) * 0.1)
        nn['nonlin'].append(activate(nonlin))
    addadadelta(nn)
    return nn


def forward(nn, data):
    """
    Given a dictionary representing a feed forward neural net and an input data matrix compute the network's output and store it within the dictionary
    :param nn: neural network dictionary
    :param data: a numpy n by m matrix where m in the number of input units in nn
    :return: the output layer activations
    """
    nn['activations'] = [data]
    nn['zs'] = []
    for w, s, b in map(None, nn['weights'], nn['nonlin'], nn['biases']):
        z = np.dot(w, nn['activations'][-1]).T + b
        nn['zs'].append(z.T)
        nn['activations'].append(s[0](z.T))
    return nn['activations'][-1]


def test_forward():
    nn = {}
    nn['eta'] = 0.1
    nn['weights'] = [np.array([[0.1, 0.2], [0.2, 0.5], [0.3, 0.4]]), np.array(np.array([0.1, 0.3, 0.2]))]
    nn['biases'] = [np.ones(3) * 0.1, np.ones(1) * 0.1]
    nn['nonlin'] = [(sigmoid, d_sigmoid)]
    x = np.array([1, 0])
    t = sigmoid(np.dot(nn['weights'][1], sigmoid(np.dot(nn['weights'][0], x) + nn['biases'][0]) + nn['biases'][1]))
    #ipdb.set_trace()
    print forward(nn, x)
    print t


def average_gradient(deltas, activations):
    dW = 0
    for i in range(deltas.shape[1]):
        dW += np.outer(deltas[:,i], activations[:,i].T)
    return dW#/deltas.shape[1]

def gradient(nn, delta):
 
    nabla_b = []
    nabla_w = []

    # output
    dact = nn['nonlin'][-1][1]
    dW = average_gradient(delta*dact(nn['zs'][-1]), nn['activations'][-2])
    nabla_b.append(np.mean(delta, axis=1))
    nabla_w.append(dW)

    for i in range(len(nn['weights']) - 2, -1, -1):
        dact = nn['nonlin'][i][1]
        delta = np.dot(nn['weights'][i+1].T, delta * dact(nn['zs'][i+1]))

        dW = average_gradient(delta,nn['activations'][i])
        nabla_b.append(np.mean(delta*dact(nn['zs'][i]),axis=1))
        nabla_w.append(dW)
    return nabla_w, nabla_b


def backprop(nn, nabla_w, nabla_b):
    eta = nn['eta']

    for i in range(len(nn['weights'])):
        nn['weights'][i] -= eta * nabla_w[-i - 1]
        nn['biases'][i] -= eta * nabla_b[-i - 1]


def expand_labels(labels):
    n = len(np.unique(labels))
    # if n == 2:
    #    l = np.array([(0,1) if row==0 else (1,0)  for row in labels]).T
    # else:
    lb = LabelBinarizer()
    l = lb.fit_transform(labels).T
    return l
def master_node(nn,data,labels):
    nabla_w = []
    nabla_b = []
    minim = len(data[0])
    maxim = len(data[0])
    newData = np.fliplr(np.asarray(data))
    newLabel = np.fliplr(np.asarray(labels))

    for i in range(minim):
      for n in range(len(nn)):
	x = nn[n]
	y = data[n][i]
	z = labels[n][i]
	
        r = forward(nn[n], data[n][i])
        delta = d_cost(r,labels[n][i])
        w,b = gradient(nn[n], delta)
        nabla_w += w
        nabla_b += b
      nabla_w = [x / len(nn) for x in nabla_w]
      nabla_b = [x / len(nn) for x in nabla_b]
      for net in nn:
        backprop(net,nabla_w,nabla_b)

def minibatch_fit(nn, data, labels):
    r = forward(nn, data)
    dact = nn['nonlin'][-1][1]
    delta = d_cost(r, labels) #* dact(nn['zs'][-1])
    backprop(nn, delta)
    # ipdb.set_trace()
    return np.sqrt(np.sum(np.square(r - labels))) / 2


def plot_decision2D(nn, data, labels, res=0.01):

    # http://stackoverflow.com/questions/19054923/plot-decision-boundary-matplotlib?answertab=votes#tab-top
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, res),
                         np.arange(y_min, y_max, res))

    Y = np.c_[xx.ravel(), yy.ravel()]
    Z = np.array([round(forward(nn,x)[0]) for x in Y])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.hold(True)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    plt.axis('off')

    # Plot also the training points
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=plt.cm.Paired)
    plt.hold(False)