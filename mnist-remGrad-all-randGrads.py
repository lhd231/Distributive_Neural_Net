import numpy as np
#from nn import nn_build, forward, master_node, plot_decision2D

from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_moons
from numpy import count_nonzero
import time
import pandas as pd

import nn_updates as nnS
#import nn_missing_grads_lilc as nnDif
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
eta = 0.01
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
    print "hi"
    print i
    print i-sample_size
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
        acc = accuracy(nn[0], val_d, val_l, thr = 0.5)
        acc1 = accuracy(nn[1], val_d, val_l, thr = 0.5)
        acc2 = accuracy(nn[2], val_d, val_l, thr = 0.5)
        l[0][test_val][c/10] = acc
        l[1][test_val][c/10] = acc1
        l[2][test_val][c/10] = acc2
        print acc
def visitClassicBatches(nn,data,labels, val_d, val_l, l, test_val, it=1000):
    for c in range(it):
      for cc in range(len(data)):
	  nnS.minibatch_fit(nn, data[cc], labels[cc])
      if c %10 == 0:
        acc = accuracy(nn, val_d, val_l, thr = 0.5)
        l[test_val][c/10] = acc
        print acc
def label_clean(label):
    nl = np.zeros(3)
    nl[label] = 1
    return nl



def accuracy(nn, data, label, thr = 0.5):
    #print "start"
    #print np.int8(nnS.forward(nn,data[1500].T) > .05)
    #print label[1500]
    acc = 0
    for i in range(len(data)):
      x = np.int8(nnS.forward(nn,data[i].T) > thr)

      if np.array_equal(x, label_clean(label[i])):
        acc += 1
    return float(acc) / float(len(data))
    #print np.int8(nnS.forward(nn,data[1500].T) > thr) == label[1500]
    #predict  = [ np.int8(nnS.forward(nn,data[c,:].T) > thr) == label_clean(label[c]) for c in range(data.shape[0])]
    #print len(np.where(np.asarray(predict)==False))
    #print np.where(np.asarray(predict)==False)[1]
    #return 100 / len(predict[0])  * np.double(len(np.where(np.asarray(predict)==False)[0]))/np.double(len(predict))

def new_acc(nn, data, label, thr = 0.5):
  predict = []
  for i in range(len(data)):
    print nnS.forward(nn,data[i])
    if np.argmax(nnS.forward(nn,data[i])) == label[i]:
      predict.append(True)
    else:
      predict.append(False)
  return 100 * np.double(len(np.where(np.array(predict) == False)[0])) / np.double(len(predict))
#Our list of accuracies.  And the number of nets (or number of sites)        
nn1Acc = [[[0 for i in range(200)]for x in range(6)] for y in range(3)]
classAcc = [[0 for i in range(200)]for x in range(6)]
eights = [[0 for i in range(200)]for x in range(6)]
sevens = [[0 for i in range(200)]for x in range(6)]
zeros = [[0 for i in range(200)]for x in range(6)]
number_of_sites = 3
sample_size = 10
def sing_run(te):
    print te
    #data, label = make_moons(n_samples=2000, shuffle=True, noise=0.01,random_state = int(time.time()))
    img_data = np.asarray(pd.read_csv("train.csv", sep=',', header=None, low_memory=False))
	# get labels
    
    
    img_data = np.delete(img_data, 0, axis=0)
    labels = np.asarray(img_data[:,0], np.dtype(int))
    img_data = np.delete(img_data, 0, axis=1)

    img_data = img_data.astype(np.float64)
    #img_data, labels = make_moons(n_samples=2000, shuffle=False, noise=0.01,random_state = 4)#int(time.time())
    total_data = [[] for i in range(3)]
    total_label = [[] for i in range(3)]
    for sas in range(len(img_data)):
      if labels[sas] == 0:
        total_data[0].append(img_data[sas])
        total_label[0].append(0)
      if labels[sas] == 3:
        total_data[0].append(img_data[sas])
        total_label[0].append(1)
      if labels[sas] == 8:
        total_data[0].append(img_data[sas])
        total_label[0].append(2)

    print "built lists"
    total_data[1] = [0]*10
    total_data[2] = [0]*10
    total_data,total_label = randomize(total_data,total_label,time.time())
    total_data[0],v,total_label[0],l = train_test_split(total_data[0],total_label[0],train_size = .3)
    validation_data = v[:1000]
    validation_label = l[:1000] 
    
    total_data[0] = total_data[0][:300]
    total_label[0] = total_label[0][:300]

    #Here, we a list of three lists for each piece of the "moon"
    #total_data, total_label = split(data,label)
    
    #total_data, total_label = randomize(total_data,total_label,4)
    
    #find the minimum between the three sides.
    minim = min(min(len(total_data[0]),len(total_data[1])),len(total_data[2]))
    print len(total_data[0])
    iters = 400

    #Our five neural networks
    nnTogetherClassic = nnS.nn_build(1,[784,20,20,3],eta=eta,nonlin="tanh")
    nnHorn1 = nnS.nn_build(1,[784,20,20,3],eta=eta,nonlin="tanh")
    nnHorn2 = nnS.nn_build(1,[784,20,20,3],eta=eta,nonlin="tanh")
    nnMiddle = nnS.nn_build(1,[784,20,20,3],eta=eta,nonlin="tanh")
    nnHorn3 = nnS.nn_build(1,[784,20,20,3],eta=eta,nonlin="tanh")
    nnHorn4 = nnS.nn_build(1,[784,20,20,3],eta=eta,nonlin="tanh")
    nnMiddle2 = nnS.nn_build(1,[784,20,20,3],eta=eta,nonlin="tanh")
    nnDecent = []
    for i in range(3):
      nnDecent.append(nnDif.nn_build(1,[784,20,20,3],eta=eta,nonlin="tanh"))
    #one = accuracy(nnTogetherClassic, validation_data, validation_label, thr=0.5)
    #total_data,total_label = randomize(total_data,total_label,time.time())
    #new_total_data,new_total_label = organize_data(total_data, total_label,minim,minim)
    horn1_data,horn1_label = [x for x in iter_minibatches(1,total_data[0][:100],total_label[0][:100])]
    horn2_data,horn2_label = [x for x in iter_minibatches(1,total_data[0][100:200],total_label[0][100:200])]
    middle_data,middle_label = [x for x in iter_minibatches(1,total_data[0][200:300],total_label[0][200:300])]
    #middle_data,middle_label = [x for x in iter_minibatches(1,total_data[0][150:200],total_label[0][150:200])]
    #middle_data,middle_label = [x for x in iter_minibatches(1,total_data[0][200:250],total_label[0][200:250])]
   # middle_data,middle_label = [x for x in iter_minibatches(1,total_data[0][250:300],total_label[0][250:300])]

    #HERE
    #total_data[1] = [0] * 10
    #total_data[2] = [0] * 10
    
    #total_data,total_label = randomize(total_data,total_label,time.time())
    centralized_data,centralized_label = [x for x in iter_minibatches(1,total_data[0],total_label[0])]
    batches_decent_data, batches_decent_label = [x for x in iter_minibatches(3,total_data[0],total_label[0])]
    print len(centralized_data)
    visitbatches(nnDecent,batches_decent_data,batches_decent_label,validation_data, validation_label,nn1Acc,te, it=iters*3)
    print "finished decents"
    #np.savetxt("decent-remGrad-all.txt",nn1Acc)
    visitClassicBatches(nnTogetherClassic,centralized_data,centralized_label,validation_data, validation_label,classAcc,te,it=iters)
    print "finished cents"
    #np.savetxt("cent-remGrad-all.txt",classAcc)
    visitClassicBatches(nnHorn1,horn1_data,horn1_label,validation_data, validation_label,eights,te,it=iters)
    print "finished horn1"
    #np.savetxt("eights-remGrad-all.txt",eights)    
    visitClassicBatches(nnHorn2,horn2_data,horn2_label,validation_data, validation_label,sevens,te,it=iters)
    #np.savetxt("sevens-remGrad-all.txt",sevens)    
    visitClassicBatches(nnMiddle,middle_data,middle_label,validation_data, validation_label,zeros,te,it=iters)
    #np.savetxt("zeros-remGrad-all.txt",zeros)

nat = range(10)
for i in range(6):
    sing_run(i)
#pool.map(sing_run,nat)
np.savetxt("decent-remGrad-all-s1-zeroedOut.txt",nn1Acc[0])
np.savetxt("decent-remGrad-all-s2-zeroedOut.txt",nn1Acc[1])
np.savetxt("decent-remGrad-all-s3-zeroedOut.txt",nn1Acc[2])
np.savetxt("cent-remGrad-all-zeroedOut.txt",classAcc)
#visitClassicBatches(nnHorn1,horn1_data,horn1_label,validation_data, validation_label,eights,te,it=iters)
print "finished horn1"
np.savetxt("eights-remGrad-all-zeroedOut.txt",eights)    
#visitClassicBatches(nnHorn2,horn2_data,horn2_label,validation_data, validation_label,sevens,te,it=iters)
np.savetxt("sevens-remGrad-all-zeroedOut.txt",sevens)    
#visitClassicBatches(nnMiddle,middle_data,middle_label,validation_data, validation_label,zeros,te,it=iters)
np.savetxt("zeros-remGrad-all-zeroedOut.txt",zeros)



