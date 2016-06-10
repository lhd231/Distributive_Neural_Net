import numpy as np
import nn as nnDif
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_moons
from numpy import count_nonzero
import time
import pylab as plt
import nnS
import math
import random
from scipy.stats import multivariate_normal as norm
from random import shuffle

minibatch = 2
nonlin = 'relu'
eta = 0.0025

def gauss(x,mu,sigma):
  return norm(mu,[[sigma,0],[0, sigma]]).pdf(x)


def split(data, label):
  midPoints = [0] * 6
  midPoints[0] = (0,1)
  midPoints[1] = (1,-.5)
  midPoints[2] = (-.9,.4)
  midPoints[3] = (0,.4)
  midPoints[4] = (1,.5)
  midPoints[5] = (2,.4)
  sectionsData = [[] for i in range(6)]
  sectionsLabel = [[] for i in range(6)]

  for x,y in zip(data,label):
    gaussResults = [[] for i in range(6)]
    for i in range(6):
      gaussResults[i] = gauss(x,midPoints[i],.1)

    a = random.random()
    s = 0
    for i in range(6):
      if y==i%2 and a < gaussResults[i]:
	sectionsData[i].append(x)
	sectionsLabel[i].append(s)
      if i % 2 == 1:
	s += 1
  
  return_list_data = list()
  return_list_label = list()
  for i in range(3):
    random.seed(i)
    arr = sectionsData[i*2] + sectionsData[i*2 + 1]
    shuffle(arr)
    return_list_data.append(arr)
    random.seed(i)
    arr = sectionsLabel[i*2] + sectionsLabel[i*2 + 1]
    shuffle(arr)
    return_list_label.append(arr)
  return return_list_data, return_list_label


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

#Calls the master node per differentiated NN.  The master node runs the forward propogation
#Then averages the returned weights and runs the back prop with the new average
def visitbatches(nn, batches, errlist, it=1000):
    for c in range(it):
        for i in range(len(batches)):
            batch = batches[i]
            cc = np.mod(c,len(batch))
            nnDif.master_node(nn, batch[cc][0], batch[cc][1])
     
#calls the forward and back propogation for a classic NN
def visitClassicBatches(nn,data, it=1000):
    for c in range(it):
        cc = np.mod(c,len(data));
        nnS.minibatch_fit(nn, data[cc][0], data[cc][1])

#Calculate the error of an differentiated NN and return it as a percent
def accuracy(nn, data, label, thr = 0.5):
    predict  = [ np.int8(nnDif.forward(nn,data[c,:]) > thr) == label[c] for c in range(data.shape[0])]
    return 100 * np.double(len(np.where(np.asarray(predict)==False)[0]))/np.double(len(predict))
  
#Calculate the error a classic NN and return as percent
def accuracyClassic(nn, data, label, thr = 0.5):
    predict  = [ np.int8(nnS.forward(nn,data[c,:]) > thr) == label[c] for c in range(data.shape[0])]
    return 100 * np.double(len(np.where(np.asarray(predict)==False)[0]))/np.double(len(predict))

def group_list(l, group_size):
    for i in xrange(0, len(l), group_size):
        yield l[i:i+group_size]
nn1Acc = [[0 for i in range(17)] for j in range(10)]
classAcc1 = [[0 for i in range(17)] for j in range(10)]
classAcc2 = [[0 for i in range(17)] for j in range(10)]
classAcc3 = [[0 for i in range(17)] for j in range(10)]
classAcc4 = [[0 for i in range(17)] for j in range(10)]
print len(nn1Acc)
print "len"
print len(nn1Acc[0])   
number_of_nets = 3

#Run the neural nets 10 times so we can find a more accurate curve
for te in range(10):
    print te
    data, label = make_moons(n_samples=1500, shuffle=True, noise=0.2,random_state = int(time.time()))
    print data[0]
    data,validation_data,label,validation_label = train_test_split(data,label,train_size = .32)
    
    #separate the data set into buckets
    total_data, total_label = split(data,label)

    
    #These are the lists of data per site.  The list contains lists of 10 data midPoints
    #We pop off the last one because it most likely does not have 10 items
    print len(total_data[0])
    print len(total_data[1])
    nn1_groups_data = list(group_list(total_data[0],10))
    nn1_groups_label = list(group_list(total_label[0],10))
    nn2_groups_data = list(group_list(total_data[1],10))
    nn2_groups_label = list(group_list(total_label[1],10))
    nn3_groups_data = list(group_list(total_data[2],10))
    nn3_groups_label = list(group_list(total_label[2],10))
    nn1_groups_data.pop()
    nn1_groups_label.pop()
    nn2_groups_data.pop()
    nn2_groups_label.pop()
    nn3_groups_data.pop()
    nn3_groups_label.pop()

    
    nets = list()  #Our differential networks
    batches = list() #a list to store every separate site set

    for i in range((min(len(nn1_groups_data),len(nn2_groups_data),len(nn2_groups_data)))):#
	
        print len(nn1_groups_data)-1
        groups_data = list()
        groups_label = list()
        nets = list()
        batches = list()
        nnClassic1 = nnS.nn_build(1,[2,6,6,1],eta=eta,nonlin=nonlin)
        nnClassic2 = nnS.nn_build(1,[2,6,6,1],eta=eta,nonlin=nonlin)
        
        for x in range(number_of_nets):
            nets.append(nnDif.nn_build(1,[2,6,6,1],eta=eta,nonlin=nonlin))
        
        #Build the two site data sets
        groups_data.append(np.asarray([item for sublist in nn1_groups_data[:i+1] for item in sublist]))
        groups_data.append(np.asarray([item for sublist in nn2_groups_data[:i+1] for item in sublist]))
        groups_data.append(np.asarray([item for sublist in nn3_groups_data[:i+1] for item in sublist]))
        print "groups data length " + str(len(nn1_groups_label))
        print "groups data length " + str(len(nn1_groups_data[i]))
        print "groups data length " + str(len(nn2_groups_label[i]))
        print "groups data length " + str(len(nn2_groups_data))

        groups_label.append(np.asarray([item for sublist in nn1_groups_label[:i+1] for item in sublist]))
        groups_label.append(np.asarray([item for sublist in nn2_groups_label[:i+1] for item in sublist]))
        groups_label.append(np.asarray([item for sublist in nn3_groups_label[:i+1] for item in sublist]))
        total_groups_data = np.asarray([item for sublist in groups_data for item in sublist])
        total_groups_label =  np.asarray([item for sublist in groups_label for item in sublist])
        #Our classic combined nn    
        nnClassic3 = nnS.nn_build(1,[2,6,6,1],eta=eta,nonlin=nonlin)
        nnClassic4 = nnS.nn_build(1,[2,6,6,1],eta=eta,nonlin=nonlin)
        
        for s in range(number_of_nets):
            batches.append([x for x in iter_minibatches(2,groups_data[s].T, groups_label[s])])
        t = [x for x in iter_minibatches(2,groups_data[0].T, groups_label[0])]
        t2 = [x for x in iter_minibatches(2,groups_data[1].T, groups_label[1])]    
        t3 = [x for x in iter_minibatches(2,groups_data[2].T, groups_label[2])]
        t4 = [x for x in iter_minibatches(2,total_groups_data.T, total_groups_label)]
        err = []
        #Run the batches through the algos
        iters = 20000
        visitClassicBatches(nnClassic1,t, it=iters)
        visitClassicBatches(nnClassic2,t2, it=iters)
        visitClassicBatches(nnClassic3,t3, it=iters)
        visitClassicBatches(nnClassic4,t4, it=iters)
        visitbatches(nets, batches, err, it=iters)
        
        #calculate error
        classic = accuracyClassic(nnClassic1,validation_data,validation_label, thr=0.5)
        one = accuracy(nets[0], validation_data, validation_label, thr=0.5)
        classic2 = accuracyClassic(nnClassic2,validation_data,validation_label, thr=0.5)
        classic3 = accuracyClassic(nnClassic3,validation_data,validation_label, thr=0.5)
        classic4 = accuracyClassic(nnClassic4,validation_data,validation_label, thr=0.5)

        #build plottable arrays
        nn1Acc[te][i] = one
        classAcc1[te][i] = classic
        classAcc2[te][i] = classic2
        classAcc3[te][i] = classic3
        classAcc4[te][i] = classic4


np.savetxt("nn1Acc-lowiters-gaussProb.txt",nn1Acc)
np.savetxt("classAcc1-lowiters-gaussProb.txt",classAcc1)
np.savetxt("classAcc2-lowiters-gaussProb.txt",classAcc2)
np.savetxt("classAcc3-lowiters-gaussProb.txt",classAcc3)
np.savetxt("classAcc-lowiters-gaussProb.txt",classAcc4)


