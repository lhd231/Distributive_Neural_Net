import numpy as np
import nnSerg as nn
from sklearn.datasets import make_moons

data, label = make_moons(n_samples=300, noise=0.4)

def iter_minibatches(chunksize, data, labels):
    # Provide chunks one by one
    chunkstartmarker = 0
    numsamples = data.shape[1]
    while chunkstartmarker < numsamples:
        chunkrows = range(chunkstartmarker,chunkstartmarker+chunksize)
        X_chunk, y_chunk = data[:,chunkrows], labels[chunkrows]
        yield X_chunk, y_chunk
        chunkstartmarker += chunksize


