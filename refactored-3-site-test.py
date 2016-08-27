import numpy as np
#from nn import nn_build, forward, master_node, plot_decision2D
import nn as nnDif
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_moons
from numpy import count_nonzero
import time

import nn_updates as nnS
import math
import random
from scipy.stats import multivariate_normal as norm
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool 
import pylab as plt
import seaborn

minibatch = 2
nonlin = 'relu'
eta = 0.001
pool = ThreadPool(10)

def gauss(x,mu,sigma):
  return norm(mu,[[sigma,0],[0, sigma]]).pdf(x)

def split(data, label):
  pointMid1 = (0,1)
  pointMid2 = (1,-.5)
  pointEdge1 = (-.9,.4)
  pointEdge2 = (.1,0)
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
  return_list_data = []
  return_list_label = []
  minim = min([len(brown1),len(brown2),len(red1),len(red2),len(green1),len(green2)])
  return_list_data.append(brown1[:minim] + brown2[:minim]) 
  return_list_data.append(green1[:minim] + green2[:minim]) 
  return_list_data.append(red1[:minim] + red2[:minim])
  return_list_label.append(brown1Label[:minim] + brown2Label[:minim]) 
  return_list_label.append(green1Label[:minim] + green2Label[:minim])
  return_list_label.append(red1Label[:minim] + red2Label[:minim])

  return return_list_data, return_list_label

#Organizes our data to make handling easier and chunks based on current position in the
#data
def organize_data(total_data,total_label,i,sample_size):
    new_total_data = []
    new_total_label = []
    for x in range(i-sample_size,i):
      new_total_data.append(total_data[0][x])
      new_total_data.append(total_data[1][x])
      new_total_data.append(total_data[2][x])
      new_total_label.append(total_label[0][x])
      new_total_label.append(total_label[1][x])
      new_total_label.append(total_label[2][x])
      #print total_label[0][x]
      #print total_label[1][x]
      #print total_label[2][x]
    return new_total_data, new_total_label
  
  
def randomize(total_data,total_label,seed):
    random.seed(seed)
    random.shuffle(total_data[0])
    random.seed(seed)
    random.shuffle(total_data[1])
    random.seed(seed)
    random.shuffle(total_data[2])
    random.seed(seed)
    random.shuffle(total_label[0])
    random.seed(seed)
    random.shuffle(total_label[1])
    random.seed(seed)
    random.shuffle(total_label[2])
    return total_data, total_label
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

    return 100 * np.double(len(np.where(np.asarray(predict)==False)[0]))/np.double(len(predict))


    
#Our list of accuracies.  And the number of nets (or number of sites)        
nn1Acc = [[0 for i in range(50)] for j in range(10)]
classAcc1 = [[0 for i in range(50)] for j in range(10)]
horn1Acc = [[0 for i in range(50)] for j in range(10)]
horn2Acc = [[0 for i in range(50)] for j in range(10)]
middleAcc = [[0 for i in range(50)] for j in range(10)]
number_of_sites = 3
sample_size = 10
def sing_run(te):
    print te
    data, label = make_moons(n_samples=2000, shuffle=True, noise=0.01,random_state = int(time.time()))
    
    data,validation_data,label,validation_label = train_test_split(data,label,train_size = .50)
    
    #Here, we a list of three lists for each piece of the "moon"
    total_data, total_label = split(data,label)
    
    total_data, total_label = randomize(total_data,total_label,4)

    #find the minimum between the three sides.
    minim = min(min(len(total_data[0]),len(total_data[1])),len(total_data[2]))

    


    #Our five neural networks
    nnTogetherClassic = nnS.nn_build(1,[2,10,10,1],eta=eta,nonlin=nonlin)
    nnHorn1 = nnS.nn_build(1,[2,10,10,1],eta=eta,nonlin=nonlin)
    nnHorn2 = nnS.nn_build(1,[2,10,10,1],eta=eta,nonlin=nonlin)
    nnMiddle = nnS.nn_build(1,[2,10,10,1],eta=eta,nonlin=nonlin)
    nnDecent = nnS.nn_build(1,[2,10,10,1],eta=eta,nonlin=nonlin)

    print minim
    data_map = []
    label_map = []
    for i in range(sample_size,minim - minim%sample_size,sample_size):#

        groups_data = []
        groups_label = []
        nets = []
        batches = []

        horn1_data,horn1_label = [x for x in iter_minibatches(1,total_data[0],total_label[0])]
	horn2_data,horn2_label = [x for x in iter_minibatches(1,total_data[2],total_label[2])]
	middle_data,middle_label = [x for x in iter_minibatches(1,total_data[1],total_label[1])]
        iters = 500
        
        #These are the total data pools.  We then turn them in mini batches for centralized and decentralized
        new_total_data,new_total_label = organize_data(total_data, total_label,i,sample_size)
        centralized_data,centralized_label = [x for x in iter_minibatches(1,new_total_data,new_total_label)]
	#print str(new_total_data[0]) + " " + str(new_total_label[0])
	#print str(new_total_data[1]) + " " + str(new_total_label[1])
	#print str(new_total_data[2]) + " " + str(new_total_label[2])
	#print str(new_total_data[3]) + " " + str(new_total_label[3])
	#print str(new_total_data[4]) + " " + str(new_total_label[4])
	#print str(new_total_data[5]) + " " + str(new_total_label[5])
	#print "break"
	batches_decent_data, batches_decent_label = [x for x in iter_minibatches(3,new_total_data,new_total_label)]
	#print str(batches_decent_data[0][0]) + " " + str(batches_decent_label[0][0])
	#print str(batches_decent_data[0][1]) + " " + str(batches_decent_label[0][1])
	#print str(batches_decent_data[0][2]) + " " + str(batches_decent_label[0][2])
	#print str(batches_decent_data[1][0]) + " " + str(batches_decent_label[1][0])
	#print str(batches_decent_data[1][1]) + " " + str(batches_decent_label[1][1])
	#print str(batches_decent_data[1][2]) + " " + str(batches_decent_label[1][2])
	data_map = data_map + batches_decent_data
	label_map = label_map + batches_decent_label
	#Debug tool to visualize data
	plotArrX = []
	plotArrY = []
	plotColor = []
	for pnt in data_map:
	  for arr in pnt:
	    plotArrX.append(arr[0])
	    plotArrY.append(arr[1])
	for l in label_map:
	  plotColor.append(l)
	#plt.scatter(plotArrX,plotArrY,c=plotColor)
	#plt.show()
	print len(batches_decent_data)
	print len(batches_decent_data[0])
	print len(batches_decent_label[0])
	print len(centralized_data)
	#Visit batches in this order:  decentralized, centralized, first horn, second horn
	# and the middle part of the crescent
	visitClassicBatches(nnDecent,batches_decent_data,batches_decent_label, it=iters)
	
	visitClassicBatches(nnTogetherClassic,centralized_data,centralized_label,it=iters)

	visitClassicBatches(nnHorn1,horn1_data[i-sample_size:i],horn1_label[i-sample_size:i],it=iters)
	visitClassicBatches(nnHorn2,horn2_data[i-sample_size:i],horn2_label[i-sample_size:i],it=iters)
	visitClassicBatches(nnMiddle,middle_data[i-sample_size:i],middle_label[i-sample_size:i],it=iters)
	
	#The accuracies of our neural networks
        togetherAcc = accuracy(nnTogetherClassic,validation_data,validation_label,thr=0.5)
        one = accuracy(nnDecent, validation_data, validation_label, thr=0.5)
	oneAcc = accuracy(nnHorn1, validation_data, validation_label, thr=0.5)
	twoAcc = accuracy(nnHorn2, validation_data, validation_label, thr=0.5)
	midAcc = accuracy(nnMiddle, validation_data, validation_label, thr=0.5)
        
        
        
	print "accuracies"
	print one
	print togetherAcc
	print oneAcc
	print twoAcc
	print midAcc

	nn1Acc[te][i/10] = one
	classAcc1[te][i/10] = togetherAcc
	horn1Acc[te][i/10] = oneAcc
	horn2Acc[te][i/10] = twoAcc
	middleAcc[te][i/10] = midAcc
nat = range(10)
#sing_run(0)
pool.map(sing_run,nat)

np.savetxt("new-3-site-decent.txt",nn1Acc)
np.savetxt("new-3-site-cent.txt",classAcc1)
np.savetxt("new-3-site-cent-horn1.txt",horn1Acc)
np.savetxt("new-3-site-cent-horn2.txt",horn2Acc)
np.savetxt("new-3-site-cent-middle.txt",middleAcc)


