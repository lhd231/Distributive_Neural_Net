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
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool 

minibatch = 2
nonlin = 'relu'
eta = 0.1
pool = ThreadPool(10)

def gauss(x,mu,sigma):
  
#  y = 1 / (math.sqrt(2.0 * math.pi * sigma) * (math.exp(((x[0] - mu[0] + 0.0) / sigma **2) + ((x[1] - mu[1] + 0.0) / sigma **2))))
  return norm(mu,[[sigma,0],[0, sigma]]).pdf(x)


def split(data, label):
  #plt.scatter(data[:,0],data[:,1], c=label)
  
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
      dif1.append(x)
      dif1Label.append(y)
    elif y==1 and a < g2:
      green2.append(x)
      green2Label.append(y)
      dif2.append(x)
      dif2Label.append(y)
    elif y==0 and a< b1:
      brown1.append(x)
      brown1Label.append(y)
      dif1.append(x)
      dif1Label.append(y)
    elif y==1 and a < b2:
      brown2.append(x)
      brown2Label.append(y)
      dif2.append(x)
      dif2Label.append(y)
    elif y==0 and a < r1:
      red1.append(x)
      red1Label.append(y)
      dif1.append(x)
      dif1Label.append(y)
    elif y==1 and a < r2:
      red2.append(x)
      red2Label.append(y)
      dif2.append(x)
      dif2Label.append(y)
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


def visitbatches(nn, batches, labels, errlist, it=1000):
  
    for c in range(it):

      nnDif.master_node(nn,batches,labels)
            #err.append(r)

def visitClassicBatches(nn,data,labels, it=1000):
    for c in range(it):
        for cc in range(len(data)):
	  nnS.minibatch_fit(nn, data[cc], labels[cc])

def accuracy(nn, data, label, thr = 0.5):

    predict  = [ np.int8(nnS.forward(nn,data[c,:]) > thr) == label[c] for c in range(data.shape[0])]
  
    return 100 * np.double(len(np.where(np.asarray(predict)==False)[0]))/np.double(len(predict))
def accuracyClassic(nn, data, label, thr = 0.5):
    predict  = [ np.int8(nnS.forward(nn,data[c,:]) > thr) == label[c] for c in range(data.shape[0])]
    return 100 * np.double(len(np.where(np.asarray(predict)==False)[0]))/np.double(len(predict))

def group_list(l, group_size):
    for i in xrange(0, len(l), group_size):
        yield np.asarray(l[i:i+group_size]).T
nn1Acc = [[0 for i in range(50)] for j in range(10)]
classAcc1 = [[0 for i in range(50)] for j in range(10)]
classAcc2 = [[0 for i in range(50)] for j in range(10)]
classAcc3 = [[0 for i in range(50)] for j in range(10)]
classAcc4 = [[0 for i in range(50)] for j in range(10)]
 
number_of_nets = 3
def sing_run(te):
    print te
    data, label = make_moons(n_samples=6000, shuffle=True, noise=0.1,random_state = int(time.time()))
    
    data,validation_data,label,validation_label = train_test_split(data,label,train_size = .40)
        #separate the data set into buckets
 
    total_data, total_label, total_dif_data, total_dif_label = split(data,label)
    
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
    totlistX = []
    totlistY = []
    for item in total_data[0]:
      totlistX.append(item[0])
      totlistY.append(item[1])
    for item in total_data[1]:
      totlistX.append(item[0])
      totlistY.append(item[1])
    for item in total_data[2]:
      totlistX.append(item[0])
      totlistY.append(item[1])
    plt.scatter(totlistX,totlistY, c=total_label[0]+total_label[1]+total_label[2])
    #random.shuffle(total_label[2])
    '''for i in range(3):
      for item in total_data[i]:
	for j in range(len(item)):
	  total_data[i][j] = list(total_data[i][j])'''
    #total_data = list(group_list(data,10))
    #total_label = list(group_list(label,10))
    #The two separate site sets
    #nn1_groups_data = total_data[:len(total_data)/2+1]
    nn1_groups_data = total_data[0]#list(group_list(total_data[0],1))
    nn1_groups_label = total_label[0]#list(group_list(total_label[0],1))
    nn2_groups_data = total_data[1]#list(group_list(total_data[1],1))
    nn2_groups_label = total_label[1]#list(group_list(total_label[1],1))
    nn3_groups_data = total_data[2]#list(group_list(total_data[2],1))
    nn3_groups_label = total_label[2]#list(group_list(total_label[2],1))
    dif_groups_data = nn1_groups_data +nn2_groups_data +nn3_groups_data
    dif_groups_label = nn1_groups_label + nn2_groups_label + nn3_groups_label
   

 
    minim = min(min(len(nn1_groups_data),len(nn2_groups_data)),len(nn3_groups_data))
    #print "length of site one group data " + str(len(nn1_groups_data))
    #nn2_groups_data = total_data[len(total_data)/2:]
    #nn1_groups_label = total_label[:len(total_data)/2+1]
    #nn2_groups_label = total_label[len(total_data)/2:]
    
    nets = list()  #Our differential networks
    batches = list() #a list to store every separate site set
    #Lists for our error to be plotted later
        
    nat = []

    dif_group_data = nn1_groups_data + nn2_groups_data + nn3_groups_data
  
    dif_group_label = nn1_groups_label + nn2_groups_label + nn3_groups_label
    differential_groups = dif_group_data[0]
    differential_labels = dif_group_label[0]
    for i in range(10,minim - minim%10,10):#

	#TODO:  Here, we need to rewrite the function so it 
        groups_data = list()
        groups_label = list()
        nets = list()
        batches = list()
        nnClassic1 = nnS.nn_build(1,[2,6,6,1],eta=eta,nonlin=nonlin)
        nnClassic2 = nnS.nn_build(1,[2,6,6,1],eta=eta,nonlin=nonlin)
        for x in range(number_of_nets):
            nets.append(nnDif.nn_build(1,[2,6,6,1],eta=eta,nonlin=nonlin))
        
        #Build the two site data sets
        #groups_data.append(np.asarray([item for sublist in nn1_groups_data[:i+1] for item in sublist]))
        #groups_data.append(np.asarray([item for sublist in nn2_groups_data[:i+1] for item in sublist]))
        #groups_data.append(np.asarray([item for sublist in nn3_groups_data[:i+1] for item in sublist]))

        #groups_label.append(np.asarray([item for sublist in nn1_groups_label[:i+1] for item in sublist]))
        #groups_label.append(np.asarray([item for sublist in nn2_groups_label[:i+1] for item in sublist]))
        #groups_label.append(np.asarray([item for sublist in nn3_groups_label[:i+1] for item in sublist]))
        #total_groups_data = np.asarray([item for sublist in groups_data for item in sublist])
        #total_groups_label =  np.asarray([item for sublist in groups_label for item in sublist])
        #Our classic combined nn    
        nnClassic3 = nnS.nn_build(1,[2,6,6,1],eta=eta,nonlin=nonlin)
        nnClassic4 = nnS.nn_build(1,[2,6,6,1],eta=eta,nonlin=nonlin)
        
        #for s in range(number_of_nets):
            #batches.append([x for x in iter_minibatches(2,groups_data[s].T, groups_label[s])])
        #t = [x for x in iter_minibatches(2,groups_data[0].T, groups_label[0])]
        #t2 = [x for x in iter_minibatches(1,groups_data[1].T, groups_label[1])]    
        #t3 = [x for x in iter_minibatches(1,groups_data[2].T, groups_label[2])]
        #t4 = [x for x in iter_minibatches(1,total_groups_data.T, total_groups_label)]
        err = []
        #Run the batches through the algos
        iters = 100
        #visitClassicBatches(nnClassic2,nn2_groups_data[:i],nn2_groups_label[:i], it=iters)
        #visitClassicBatches(nnClassic3,nn3_groups_data[:i],nn3_groups_label[:i], it=iters)
        #visitClassicBatches(nnClassic4,t4, it=iters)
        #visitbatches(nets, batches, err, it=iters)
	#print "finish classics"
        #differential_groups = differential_groups + dif_group_data[3*i] +dif_group_data[3*i + 1] + dif_group_data[3*i + 2]
	batchesData1,batchesLabel1 = [x for x in iter_minibatches(1,nn1_groups_data,nn1_groups_label)]
       	batchesData2,batchesLabel2 = [x for x in iter_minibatches(1,nn2_groups_data,nn2_groups_label)]
	batchesData3,batchesLabel3 = [x for x in iter_minibatches(1,nn3_groups_data,nn3_groups_label)]
	nnTogetherClassic = nnS.nn_build(1,[2,6,6,1],eta=eta,nonlin=nonlin)
	nnTogetherDif = nnDif.nn_build(1,[2,6,6,1],eta=eta,nonlin=nonlin)
	visitClassicBatches(nnTogetherClassic,batchesData1[:i]+batchesData2[:i]+batchesData3[:i],batchesLabel1[:i]+batchesLabel2[:i]+batchesLabel3[:i],it=iters)
	visitClassicBatches(nnClassic1,batchesData1[:i],batchesLabel1[:i], it=iters)
	visitClassicBatches(nnClassic2,batchesData2[:i],batchesLabel2[:i], it=iters)
	visitClassicBatches(nnClassic3,batchesData3[:i],batchesLabel3[:i], it=iters)
	
	plotlistX = []
	plotlistY = []
	plotlistColor = []
	'''for batch in batchesData1[:i]:
	  for tup in batch:
	    for sid in tup:
	      plotlistX.append(sid[0])
	      plotlistY.append(sid[1])
	for batch in batchesData2[:i]:
	  for tup in batch:
	    for sid in tup:
	      plotlistX.append(sid[0])
	      plotlistY.append(sid[1])
	for batch in batchesData3[:i]:
	  for tup in batch:
	    x = 2
	    #plotlistX.append(tup[0])
	    #plotlistY.append(tup[1])
	for arr in batchesLabel1[:i]:
	  for tup in arr:
	    plotlistColor.append(tup[0])
	    plotlistColor.append(tup[0])
	for arr in batchesLabel2[:i]:
	  for tup in arr:
	    plotlistColor.append(tup[0])
	    plotlistColor.append(tup[0])

	#plt.scatter(plotlistX,plotlistY,c=plotlistColor)
	     #+batchesLabel2[:i]+batchesLabel3[:i])
	
	#plt.show()'''
	
        #differential_labels = differential_labels + dif_group_label[3*i] + dif_group_label[3*i + 1] + dif_group_label[3*i + 2]
        #visitbatches(nets, [nn1_groups_data[:i],nn2_groups_data[:i],nn3_groups_data[:i]], [nn1_groups_label[:i],nn2_groups_label[:i],nn3_groups_label[:i]], err, it=iters)
	visitbatches(nets, [batchesData1[:i],batchesData2[:i],batchesData3[:i]], [batchesLabel1[:i],batchesLabel2[:i],batchesLabel3[:i]], err, it=iters)
        #visitbatches([nets[1]], [batchesData1[:i]], [batchesLabel1[:i]], err, it=iters)
	#visitbatches([nnTogetherDif], [batchesData1[:i]+batchesData2[:i]], [batchesLabel1[:i]+batchesLabel2[:i]], err, it=iters)

        #calculate error
        nnClassic5 = nnS.nn_build(1,[2,6,6,1],eta=eta,nonlin=nonlin)
        #testAcc = accuracyClassic(nnClassic5,validation_data,validation_label,thr=.05)
        #testAcc2 = accuracy(nnClassic5,validation_data,validation_label,thr=.05)
        togetherAcc = accuracy(nnTogetherClassic,validation_data,validation_label,thr=.05)
        
        
        classic = accuracyClassic(nnClassic1,validation_data,validation_label, thr=0.5)
        one = accuracy(nets[0], validation_data, validation_label, thr=0.5)
        #two = accuracy(nets[1], validation_data,validation_label, thr=0.5)
        classic2 = accuracyClassic(nnClassic2,validation_data,validation_label, thr=0.5)
        classic3 = accuracyClassic(nnClassic3,validation_data,validation_label, thr=0.5)
        classic4 = accuracyClassic(nnClassic4,validation_data,validation_label, thr=0.5)
        #build plottable arrays
  
        nn1Acc[te][i/10] = one
	newAcc = accuracy(nnTogetherDif,validation_data,validation_label,thr=.5)
        print "ACCURACY"
        print one
        print "^^^Decent acc  || total acc"
        print togetherAcc
        print "new acc  " + str(classic)
      
        classAcc1[te][i/10] = classic
        classAcc2[te][i/10] = classic2
        classAcc3[te][i/10] = classic3
        classAcc4[te][i/10] = togetherAcc

        #print "us " + str(one) + " c1 " + str(classic) + " c2 " + str(classic2) + " cc " + str(classic3)
nat = range(10)
sing_run(3)
#pool.map(sing_run,nat)
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

