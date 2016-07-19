import numpy as np
#from nn import nn_build, forward, master_node, plot_decision2D
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

minibatch = 2
nonlin = 'relu'
eta = 0.0025

def gauss(x,mu,sigma):
  
#  y = 1 / (math.sqrt(2.0 * math.pi * sigma) * (math.exp(((x[0] - mu[0] + 0.0) / sigma **2) + ((x[1] - mu[1] + 0.0) / sigma **2))))
  return norm(mu,[[sigma,0],[0, sigma]]).pdf(x)


def split(data, label):
  #plt.scatter(data[:,0],data[:,1], c=label)
  
  print np.var(data)
  pointMid1 = (0,1)
  pointMid2 = (1,-.5)
  pointEdge1 = (-.9,.4)
  pointEdge2 = (0,.4)
  pointEdge3 = (1,.5)
  pointEdge4 = (2,.4)
  brown1 = list()
  brown2 = list()
  green1 = list()
  green2 = list()
  red1 = list()
  red2 = list()
  brown1Label = list()
  brown2Label = list()
  green1Label = list()
  green2Label = list()
  red1Label = list()
  red2Label = list()
  max = 0
  i = 0
  for x,y in zip(data,label):
    b1 = gauss(x,pointMid1,.1)
    b2 = gauss(x,pointMid2,.1)
    g1 = gauss(x,pointEdge1,.1)
    g2 = gauss(x,pointEdge2,.1)
    r1 = gauss(x,pointEdge3,.1)
    r2 = gauss(x,pointEdge4,.1)
    a = random.random()
    if y==0 and a < g1:
      green1.append(x)
      green1Label.append(y)
    elif y==1 and a < g2:
      green2.append(x)
      green2Label.append(y)
    elif y==0 and a< b1:
      brown1.append(x)
      brown1Label.append(y)
    elif y==1 and a < b2:
      brown2.append(x)
      brown2Label.append(y)

    elif y==0 and a < r1:
      red1.append(x)
      red1Label.append(y)
    elif y==1 and a < r2:
      red2.append(x)
      red2Label.append(y)
    i += 1
  return_list_data = list()
  return_list_label = list()
  return_list_data.append(brown1 + brown2) 
  return_list_data.append(green1 + green2) 
  return_list_data.append(red1 + red2)
  return_list_label.append(brown1Label + brown2Label) 
  return_list_label.append(green1Label + green2Label)
  return_list_label.append(red1Label + red2Label)
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


def visitbatches(nn, batches, labels, errlist, it=1000):
    for c in range(it):
      print 'HERE BE BATCHES   ' + str(batches)
      print "HERE BE LABELS    " + str(labels)
      print "HERE BE NN LEN   " + str(len(nn))
      nnDif.master_node(nn,batches,labels)
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
        yield np.asarray(l[i:i+group_size])
nn1Acc = [[0 for i in range(17)] for j in range(10)]
classAcc1 = [[0 for i in range(17)] for j in range(10)]
classAcc2 = [[0 for i in range(17)] for j in range(10)]
classAcc3 = [[0 for i in range(17)] for j in range(10)]
classAcc4 = [[0 for i in range(17)] for j in range(10)]
print len(nn1Acc)
print "len"
print len(nn1Acc[0])   
number_of_nets = 3
for te in range(10):
    print te
    data, label = make_moons(n_samples=1500, shuffle=True, noise=0.2,random_state = int(time.time()))
    print data[0]
    data,validation_data,label,validation_label = train_test_split(data,label,train_size = .32)
        #separate the data set into buckets
    total_data, total_label = split(data,label)

     
    #total_data = list(group_list(data,10))
    #total_label = list(group_list(label,10))
    #The two separate site sets
    #nn1_groups_data = total_data[:len(total_data)/2+1]
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
    #print "length of site one group data " + str(len(nn1_groups_data))
    #nn2_groups_data = total_data[len(total_data)/2:]
    #nn1_groups_label = total_label[:len(total_data)/2+1]
    #nn2_groups_label = total_label[len(total_data)/2:]
    
    nets = list()  #Our differential networks
    batches = list() #a list to store every separate site set
    #Lists for our error to be plotted later
        
    nat = []
    for i in range((min(len(nn1_groups_data),len(nn2_groups_data),len(nn2_groups_data)))):#
	#TODO:  Here, we need to rewrite the function so it 
        print "HERE IS OUR i    " + str(i)
        groups_data = list()
        groups_label = list()
        nets = list()
        batches = list()
        nnClassic1 = nnS.nn_build(1,[2,6,6,1],eta=eta,nonlin=nonlin)
        nnClassic2 = nnS.nn_build(1,[2,6,6,1],eta=eta,nonlin=nonlin)
        print "HERE BE GROUP1 DATA   " 
        print nn1_groups_data
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
        t2 = [x for x in iter_minibatches(1,groups_data[1].T, groups_label[1])]    
        t3 = [x for x in iter_minibatches(1,groups_data[2].T, groups_label[2])]
        t4 = [x for x in iter_minibatches(1,total_groups_data.T, total_groups_label)]
        err = []
        #Run the batches through the algos
        iters = 20000
        #visitClassicBatches(nnClassic1,t, it=iters)
        #visitClassicBatches(nnClassic2,t2, it=iters)
        #visitClassicBatches(nnClassic3,t3, it=iters)
        #visitClassicBatches(nnClassic4,t4, it=iters)
        #visitbatches(nets, batches, err, it=iters)
        differential_groups = []
        differential_groups.append(groups_data[3*i])
        differential_groups.append(groups_data[3*i + 1])
        differential_groups.append(groups_data[3*i + 2])
        differential_labels = []
        differential_labels.append(groups_label[3*i])
        differential_labels.append(groups_label[3*i + 1])
        differential_labels.append(groups_label[3*i + 2])
        visitbatches(nets, differential_groups, differential_labels, err, it=iters)
        #calculate error
        classic = accuracyClassic(nnClassic1,validation_data,validation_label, thr=0.5)
        one = accuracy(nets[0], validation_data, validation_label, thr=0.5)
        classic2 = accuracyClassic(nnClassic2,validation_data,validation_label, thr=0.5)
        classic3 = accuracyClassic(nnClassic3,validation_data,validation_label, thr=0.5)
        classic4 = accuracyClassic(nnClassic4,validation_data,validation_label, thr=0.5)
        nat = nets[0]
        #build plottable arrays
        nn1Acc[te][i] = one
        classAcc1[te][i] = classic
        classAcc2[te][i] = classic2
        classAcc3[te][i] = classic3
        classAcc4[te][i] = classic4

        #print "us " + str(one) + " c1 " + str(classic) + " c2 " + str(classic2) + " cc " + str(classic3)

#nn1Acc[:] = [x / 10 for x in nn1Acc]
#classAcc1[:] = [x / 10 for x in classAcc1]
#classAcc2[:] = [x / 10 for x in classAcc2]
#classAcc3[:] = [x / 10 for x in classAcc3]
np.savetxt("nn1Acc-lowiters-gaussProb.txt",nn1Acc)
np.savetxt("classAcc1-lowiters-gaussProb.txt",classAcc1)
np.savetxt("classAcc2-lowiters-gaussProb.txt",classAcc2)
np.savetxt("classAcc3-lowiters-gaussProb.txt",classAcc3)
np.savetxt("classAcc-lowiters-gaussProb.txt",classAcc4)
#plt.xlabel("Number of batches [of size 50]")
#plt.ylabel("Error rate")

#plt.plot(nn1Acc[0], "-r", label = "differential")
#plt.plot(nn1Acc[1], "-r", label = "differential")
#plt.plot(classAcc2, "-b", label = "classic 1")
#plt.plot(classAcc1, "-g", label = "classic 2")
#plt.plot(classAcc3, "-p", label = "Classic Combined")
#plt.legend(loc='upper right')

