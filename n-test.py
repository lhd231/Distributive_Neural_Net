import numpy as np
#from nn import nn_build, forward, master_node, plot_decision2D
import nn as nnDif
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_moons
import pylab as plt
import nnS
minibatch = 2
nonlin = 'relu'
eta = 0.0025

data, label = make_moons(n_samples=1000, noise=0.05)

#adagrad and adadelta  esp adadelta
def iter_minibatches(chunksize, data, labels):
    # Provide chunks one by one
    chunkstartmarker = 0
    numsamples = data.shape[1]
    while chunkstartmarker < numsamples:
        chunkrows = range(chunkstartmarker,chunkstartmarker+chunksize)
        X_chunk, y_chunk = data[:,chunkrows], labels[chunkrows]
        yield X_chunk, y_chunk
        chunkstartmarker += chunksize


def visitbatches(nn, batches, errlist, it=1000):
    for c in range(it):
        for i in range(len(batches)):
            batch = batches[i]
            cc = np.mod(c,len(batch))
            nnDif.master_node(nn, batch[cc][0], batch[cc][1])
            #err.append(r)

def visitClassicBatches(nn,data, it=1000):
    for c in range(it):
        cc = np.mod(c,len(data));
        nnS.minibatch_fit(nn, data[cc][0], data[cc][1])

def accuracy(nn, data, label, thr = 0.5):
    predict  = [ np.int8(nnDif.forward(nn,data[c,:]) > thr) == label[c] for c in range(data.shape[0])]
    return 100 * np.double(len(np.where(np.asarray(predict)==False)[0]))/np.double(len(predict))
def accuracyClassic(nn, data, label, thr = 0.5):
    predict  = [ np.int8(nnS.forward(nn,data[c,:]) > thr) == label[c] for c in range(data.shape[0])]
    return 100 * np.double(len(np.where(np.asarray(predict)==False)[0]))/np.double(len(predict))

def group_list(l, group_size):
    for i in xrange(0, len(l), group_size):
        yield l[i:i+group_size]
       
number_of_nets = 2
data,validation_data,label,validation_label = train_test_split(data,label,train_size = .32)
#separate the data set into buckets
print 
total_data = list(group_list(data,10))
total_label = list(group_list(label,10))

#The two separate site sets
nn1_groups_data = total_data[:len(total_data)/2+1]
nn2_groups_data = total_data[len(total_data)/2:]
nn1_groups_label = total_label[:len(total_data)/2+1]
nn2_groups_label = total_label[len(total_data)/2:]

nets = list()  #Our differential networks
batches = list() #a list to store every separate site set
#Lists for our error to be plotted later
nn1Acc = list()
classAcc1 = list()
classAcc2 = list()
classAcc3 = list()
nat = []
for i in range(len(nn1_groups_data)/2):
    groups_data = list()
    groups_label = list()
    nets = list()
    batches = list()
    nnClassic1 = nnS.nn_build(1,[2,6,6,1],eta=eta,nonlin=nonlin)
    nnClassic2 = nnS.nn_build(1,[2,6,6,1],eta=eta,nonlin=nonlin)

    for x in range(number_of_nets):
        nets.append(nnDif.nn_build(1,[2,6,6,1],eta=eta,nonlin=nonlin))

    #Build the two site data sets
    groups_data.append(np.asarray([item for sublist in nn1_groups_data[:i*2+1] for item in sublist]))
    groups_data.append(np.asarray([item for sublist in nn2_groups_data[:i*2+1] for item in sublist]))
  
    groups_label.append(np.asarray([item for sublist in nn1_groups_label[:i*2+1] for item in sublist]))
    groups_label.append(np.asarray([item for sublist in nn2_groups_label[:i*2+1] for item in sublist]))
      
    total_groups_data = np.asarray([item for sublist in groups_data for item in sublist])
    total_groups_label =  np.asarray([item for sublist in groups_label for item in sublist])
    #Our classic combined nn    
    nnClassic3 = nnS.nn_build(1,[2,6,6,1],eta=eta,nonlin=nonlin)
    print len(nn1_groups_data[:i*2+1])
    print len(groups_data[0])
    for i in range(number_of_nets):
        batches.append([x for x in iter_minibatches(1,groups_data[i].T, groups_label[i])])
    t = [x for x in iter_minibatches(2,groups_data[0].T, groups_label[0])]
    t2 = [x for x in iter_minibatches(2,groups_data[1].T, groups_label[1])]    
    t3 = [x for x in iter_minibatches(1,total_groups_data.T, total_groups_label)]
    err = []
    #Run the batches through the algos
    visitClassicBatches(nnClassic1,t)
    visitClassicBatches(nnClassic2,t2)
    visitClassicBatches(nnClassic3,t3, it=100)
    visitbatches(nets, batches, err, it=100)
    
    #calculate error
    classic = accuracyClassic(nnClassic1,validation_data,validation_label, thr=0.5)
    one = accuracy(nets[0], validation_data, validation_label, thr=0.5)
    classic2 = accuracyClassic(nnClassic2,validation_data,validation_label, thr=0.5)
    classic3 = accuracyClassic(nnClassic3,validation_data,validation_label, thr=0.5)
    nat = nets[0]
    #build plottable arrays
    nn1Acc.append(one)
    classAcc1.append(classic)
    classAcc2.append(classic2)
    classAcc3.append(classic3)
plt.xlabel("Number of batches [of size 50]")
plt.ylabel("Error rate")

plt.plot(nn1Acc, "-r", label = "differential")
plt.plot(classAcc2, "-b", label = "classic 1")
plt.plot(classAcc1, "-g", label = "classic 2")
plt.plot(classAcc3, "-p", label = "Classic Combined")
plt.legend(loc='upper right')
