import numpy as np
#from nn import nn_build, forward, master_node, plot_decision2D
import nn as nnDif
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_moons
from numpy import count_nonzero
import time

import nnS
import math
import random
from scipy.stats import multivariate_normal as norm
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool 
import pylab as plt
import seaborn

minibatch = 2
nonlin = 'relu'
eta = 0.025
pool = ThreadPool(10)

def gauss(x,mu,sigma):
  return norm(mu,[[sigma,0],[0, sigma]]).pdf(x)

def split(data, label):
  pointMid1 = (0,1)
  pointMid2 = (1,-.5)
  pointEdge1 = (-.9,.4)
  pointEdge2 = (0,.4)
  pointEdge3 = (.8,.4)
  pointEdge4 = (1.8,0)
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
  dif1 = list()
  dif2 = list()
  dif1Label = list()
  dif2Label = list()
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
  return_dif_data = list()
  return_dif_label = list()
  minim = min([len(brown1),len(brown2),len(red1),len(red2),len(green1),len(green2)])
  return_list_data.append(brown1[:minim] + brown2[:minim]) 
  return_list_data.append(green1[:minim] + green2[:minim]) 
  return_list_data.append(red1[:minim] + red2[:minim])
  return_list_label.append(brown1Label[:minim] + brown2Label[:minim]) 
  return_list_label.append(green1Label[:minim] + green2Label[:minim])
  return_list_label.append(red1Label[:minim] + red2Label[:minim])
  return_dif_data.append(dif1 + dif2)
  return_dif_label.append(dif1Label + dif2Label)
  rads = brown1[:minim] + brown2[:minim]
  list1, list2 = zip(*rads)
  return return_list_data, return_list_label, return_dif_data, return_dif_label



#adagrad and adadelta  esp adadelta
def iter_minibatches(chunksize, data, labels):
    # Provide chunks one by one
    chunkstartmarker = 0
    numsamples = len(data)
    X_chunk = []
    Y_chunk = []
    while chunkstartmarker < numsamples:
        chunkrows = range(chunkstartmarker,chunkstartmarker+chunksize)
        X_chunk.append(np.asarray(data[chunkstartmarker:chunkstartmarker+chunksize]))
        Y_chunk.append(np.asarray(labels[chunkstartmarker:chunkstartmarker+chunksize]))
  
        chunkstartmarker += chunksize
    return X_chunk, Y_chunk


def visitbatches(nn, batches, labels, it=1000):
    for c in range(it):
      nnDif.master_node(nn,batches,labels)
            #err.append(r)

def visitClassicBatches(nn,data,labels, it=1000):
    for c in range(it):
        for cc in range(len(data)):
	  nnS.minibatch_fit(nn, data[cc], labels[cc])

def accuracy(nn, data, label, thr = 0.5):
    predict  = [ np.int8(nnDif.forward(nn,data[c,:]) > thr) == label[c] for c in range(data.shape[0])]
    plotArrX = []
    plotArrY = []
    for pnt in data:
      plotArrX.append(pnt[0])
      plotArrY.append(pnt[1])
    #plt.scatter(plotArrX,plotArrY,c=predict)
    #plt.show()
    return 100 * np.double(len(np.where(np.asarray(predict)==False)[0]))/np.double(len(predict))


def group_list(l, group_size):
    for i in xrange(0, len(l), group_size):
        yield np.asarray(l[i:i+group_size]).T
        
        
nn1Acc = [[0 for i in range(50)] for j in range(10)]
classAcc1 = [[0 for i in range(50)] for j in range(10)]
horn1Acc = [[0 for i in range(50)] for j in range(10)]
horn2Acc = [[0 for i in range(50)] for j in range(10)]
middleAcc = [[0 for i in range(50)] for j in range(10)]
number_of_nets = 3

def sing_run(te):
    print te
    data, label = make_moons(n_samples=2000, shuffle=True, noise=0.1,random_state = int(time.time()))
    
    data,validation_data,label,validation_label = train_test_split(data,label,train_size = .38)
        #separate the data set into buckets
 
    total_data, total_label,A,B = split(data,label)
   
    random.seed(4)
    random.shuffle(total_data[0])
    random.seed(4)
    random.shuffle(total_data[1])
    random.seed(4)
    random.shuffle(total_data[2])
    random.seed(4)
    random.shuffle(total_label[0])
    random.seed(4)
    random.shuffle(total_label[1])
    random.seed(4)
    random.shuffle(total_label[2])


   

    #find the minimum between the three sides.
    minim = min(min(len(total_data[0]),len(total_data[1])),len(total_data[2]))

    
    nets = list()  #Our differential networks




  

    for i in range(10,minim - minim%10,10):#

        groups_data = []
        groups_label = []
        nets = []
        batches = []
	plotlistX = []
	plotlistY = []
	plotlistColor = []
	for batch in total_data[0][:i]:

	      plotlistX.append(batch[0])
	      plotlistY.append(batch[1])
	for batch in total_data[1][:i]:
	 
	
	      plotlistX.append(batch[0])
	      plotlistY.append(batch[1])
	for batch in total_data[2][:i]:
	
	    x = 2
	    plotlistX.append(batch[0])
	    plotlistY.append(batch[1])
	for arr in total_label[0][:i]:
	  
	    plotlistColor.append([arr,arr,arr])
	for arr in total_label[1][:i]:
	  
	    plotlistColor.append([arr,arr,arr])
	for arr in total_label[2][:i]:
	
	    plotlistColor.append([arr,arr,arr])
	#print "gogo"
	#plt.ylabel("Error rate")
	#plt.xlabel("Group size (3 groups per decentralized line")
	#plt.scatter(plotlistX,plotlistY,c=plotlistColor)
	#plt.show()
	
        nnTogetherClassic = nnS.nn_build(1,[2,6,6,1],eta=eta,nonlin=nonlin)
        nnHorn1 = nnS.nn_build(1,[2,6,6,1],eta=eta,nonlin=nonlin)
        nnHorn2 = nnS.nn_build(1,[2,6,6,1],eta=eta,nonlin=nonlin)
        nnMiddle = nnS.nn_build(1,[2,6,6,1],eta=eta,nonlin=nonlin)
        for x in range(number_of_nets):
            nets.append(nnDif.nn_build(1,[2,6,6,1],eta=eta,nonlin=nonlin))


        iters = 700
        new_total_data = []
        new_total_label = []
        for x in range(i):
	  new_total_data.append(total_data[0][x])
	  new_total_data.append(total_data[1][x])
	  new_total_data.append(total_data[2][x])
	  new_total_label.append(total_label[0][x])
	  new_total_label.append(total_label[1][x])
	  new_total_label.append(total_label[2][x])
        batchesData1,batchesLabel1 = [x for x in iter_minibatches(1,total_data[0],total_label[0])]
       	batchesData2,batchesLabel2 = [x for x in iter_minibatches(1,total_data[1],total_label[1])]
	batchesData3,batchesLabel3 = [x for x in iter_minibatches(1,total_data[2],total_label[2])]
	
	batches_decent_data, batches_decent_label = [x for x in iter_minibatches(3,new_total_data,new_total_label)]
	visitClassicBatches(nets[0],batches_decent_data,batches_decent_label, it=iters)
	#print "finished decent"
	visitClassicBatches(nnTogetherClassic,batchesData1[:i]+batchesData2[:i]+batchesData3[:i],batchesLabel1[:i]+batchesLabel2[:i]+batchesLabel3[:i],it=iters)
	visitClassicBatches(nnHorn1,batchesData1[:i],batchesLabel1[:i],it=iters)
	visitClassicBatches(nnHorn2,batchesData3[:i],batchesLabel3[:i],it=iters)
	visitClassicBatches(nnMiddle,batchesData2[:i],batchesLabel2[:i],it=iters)
	#visitbatches(nets, [batchesData1[:i],batchesData2[:i],batchesData3[:i]], [batchesLabel1[:i],batchesLabel2[:i],batchesLabel3[:i]], it=iters)

        togetherAcc = accuracy(nnTogetherClassic,validation_data,validation_label,thr=.05)
        
        
        one = accuracy(nets[0], validation_data, validation_label, thr=0.5)
	oneAcc = accuracy(nnHorn1, validation_data, validation_label, thr=0.5)
	twoAcc = accuracy(nnHorn2, validation_data, validation_label, thr=0.5)
	midAcc = accuracy(nnHorn2, validation_data, validation_label, thr=0.5)
        nn1Acc[te][i/10] = one
	print "accuracies"
	print one
        classAcc1[te][i/10] = togetherAcc
	print togetherAcc
	print oneAcc
	print twoAcc
	print midAcc

	horn1Acc[te][i/10] = oneAcc
	horn2Acc[te][i/10] = twoAcc
	middleAcc[te][i/10] = midAcc
nat = range(10)
#sing_run(0)
pool.map(sing_run,nat)

np.savetxt("3-site-decent-2.txt",nn1Acc)
np.savetxt("3-site-cent-2.txt",classAcc1)
np.savetxt("3-site-cent-horn1-2.txt",horn1Acc)
np.savetxt("3-site-cent-horn2-2.txt",horn2Acc)
np.savetxt("3-site-cent-middle-2.txt",middleAcc)


