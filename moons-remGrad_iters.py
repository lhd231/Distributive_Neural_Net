import numpy as np
#from nn import nn_build, forward, master_node, plot_decision2D

from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_moons
from numpy import count_nonzero
import time
import pandas as pd

import nn_updates as nnS
import nn_missing_grads_zeroedOut as nnDif
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


def visitbatches(nn,data,labels, val_d, val_l, l, test_val, it=1000):
    for c in range(it):
      for cc in range(len(data)):
         nnDif.master_node(nn, data[cc], labels[cc])
      if c %10 == 0:
        acc1 = accuracy(nn[0], val_d, val_l, thr = 0.5)
        acc2 = accuracy(nn[1], val_d, val_l, thr = 0.5)
        acc3 = accuracy(nn[2], val_d, val_l, thr = 0.5)
        l[0][test_val][c/10] = acc1
        l[1][test_val][c/10] = acc2
        l[2][test_val][c/10] = acc3

        print acc2

def visitClassicBatches(nn,data,labels, val_d, val_l, l, test_val, it=1000):
    for c in range(it):
      for cc in range(len(data)):
	  nnS.minibatch_fit(nn, data[cc], labels[cc])
      if c %10 == 0:
        acc = accuracy(nn, val_d, val_l, thr = 0.5)
        print test_val
        l[test_val][c/10] = acc
        print acc
def label_clean(label):
    nl = np.zeros(3)
    nl[label] = 1
    return nl



def accuracy(nn, data, label, thr = 0.5):
    predict  = [ np.int8(nnDif.forward(nn,data[c,:]) > thr) == label[c] for c in range(data.shape[0])]

    return 100 * np.double(len(np.where(np.asarray(predict)==False)[0]))/np.double(len(predict))

def new_acc(nn, data, label, thr = 0.5):
  predict = []
  for i in range(len(data)):
    if np.argmax(nnS.forward(nn,data[i])) == label[i]:
      predict.append(True)
    else:
      predict.append(False)
  return 100 * np.double(len(np.where(np.array(predict) == False)[0])) / np.double(len(predict))
#Our list of accuracies.  And the number of nets (or number of sites)
iters = 2000        
nn1Acc = [[[0 for i in range(iters/10)]for x in range(10)] for y in range(3)]
classAcc = [[0 for i in range(iters/10)]for x in range(10)]
eights = [[0 for i in range(iters/10)]for x in range(10)]
sevens = [[0 for i in range(iters/10)]for x in range(10)]
zeros = [[0 for i in range(iters/10)]for x in range(10)]
number_of_sites = 3
sample_size = 20
def sing_run(te):
    print te
    data, label = make_moons(n_samples=2000, shuffle=True, noise=0.01,random_state = int(time.time()))
    
    data,validation_data,label,validation_label = train_test_split(data,label,train_size = .50)
    
    #Here, we a list of three lists for each piece of the "moon"
    total_data, total_label = split(data,label)
    
    total_data, total_label = randomize(total_data,total_label,time.time())
    #for item in validation_label:
      #print item
    #exit(1)
    
    #HERE    
    
    print len(total_data)
    print len(total_data[0])
    horn1_data,horn1_label = [x for x in iter_minibatches(1,total_data[0][:sample_size],total_label[0])]
    horn2_data,horn2_label = [x for x in iter_minibatches(1,total_data[2][:sample_size],total_label[2])]
    middle_data,middle_label = [x for x in iter_minibatches(1,total_data[1][:sample_size],total_label[1])]
    iters = 500
        
    #These are the total data pools.  We then turn them in mini batches for centralized and decentralized
    new_total_data,new_total_label = organize_data(total_data, total_label,sample_size,sample_size)
    print len(new_total_data)
    centralized_data,centralized_label = [x for x in iter_minibatches(1,new_total_data,new_total_label)]
    #print str(new_total_data[0]) + " " + str(new_total_label[0])
    #print str(new_total_data[1]) + " " + str(new_total_label[1])
    #print str(new_total_data[2]) + " " + str(new_total_label[2])
    #print str(new_total_data[3]) + " " + str(new_total_label[3])
    #print str(new_total_data[4]) + " " + str(new_total_label[4])
    #print str(new_total_data[5]) + " " + str(new_total_label[5])
    #print "break"
    batches_decent_data, batches_decent_label = [x for x in iter_minibatches(3,new_total_data,new_total_label)]
    

    #Here, we a list of three lists for each piece of the "moon"
    #total_data, total_label = split(data,label)
    
    #total_data, total_label = randomize(total_data,total_label,4)
    
    #find the minimum between the three sides.
    
    minim = min(min(len(total_data[0]),len(total_data[1])),len(total_data[2]))
    print minim


    #Our five neural networks
    nnTogetherClassic = nnS.nn_build(1,[2,6,6,1],eta=eta,nonlin="tanh")
    nnHorn1 = nnS.nn_build(1,[2,6,6,1],eta=eta,nonlin="tanh")
    nnHorn2 = nnS.nn_build(1,[2,6,6,1],eta=eta,nonlin="tanh")
    nnMiddle = nnS.nn_build(1,[2,6,6,1],eta=eta,nonlin="tanh")
    nnDecent = []
    for i in range(3):
      nnDecent.append(nnS.nn_build(1,[2,6,6,1],eta=eta,nonlin="tanh"))
    #one = accuracy(nnTogetherClassic, validation_data, validation_label, thr=0.5)
    total_data, total_label = randomize(total_data,total_label,time.time())
    #new_total_data,new_total_label = organize_data(total_data, total_label,minim,minim)
   
    #HERE
    
    #centralized_data,centralized_label = [x for x in iter_minibatches(1,total_data[0]+total_data[1]+total_data[2],total_label[0]+total_label[1]+total_label[2])]
    batches_decent_data, batches_decent_label = [x for x in iter_minibatches(3,new_total_data,new_total_label)]
    print len(batches_decent_data[0])
    visitbatches(nnDecent,batches_decent_data,batches_decent_label,validation_data, validation_label,nn1Acc,te, it=iters)
    print "finished decents"

    visitClassicBatches(nnTogetherClassic,centralized_data,centralized_label,validation_data, validation_label,classAcc,te,it=iters)
    print "finished cents"
    
    visitClassicBatches(nnHorn1,horn1_data,horn1_label,validation_data, validation_label,eights,te,it=iters)
    print "finished horn1"
      
    visitClassicBatches(nnHorn2,horn2_data,horn2_label,validation_data, validation_label,sevens,te,it=iters)
        
    visitClassicBatches(nnMiddle,middle_data,middle_label,validation_data, validation_label,zeros,te,it=iters)
    

nat = range(10)
for i in range(6):
    sing_run(i)
#pool.map(sing_run,nat)

np.savetxt("decent-remGrad-s1-moons-noisy.txt",nn1Acc[0])
np.savetxt("decent-remGrad-s2-moons-noisy.txt",nn1Acc[1])
np.savetxt("decent-remGrad-s3-moons-noisy.txt",nn1Acc[2])
np.savetxt("cent-remGrad-moons-noisy.txt",classAcc)
np.savetxt("eights-remGrad-moons-noisy.txt",eights)  
np.savetxt("sevens-remGrad-moons-noisy.txt",sevens)
np.savetxt("zeros-remGrad-moons-noisy.txt",zeros)
