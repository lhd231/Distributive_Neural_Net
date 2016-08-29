import numpy as np
#from nn import nn_build, forward, master_node, plot_decision2D
import nn as nnDif
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_moons
import time
import pylab as plt
import nnS
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool 


minibatch = 2
nonlin = 'relu'
eta = 0.0025
pool = ThreadPool(6)

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
    
    return np.asarray(X_chunk), np.asarray(Y_chunk)

def visitbatches(nn, batches, labelBatches, errlist, it=1000):
    for c in range(it):
	    #print "len of things " + str(len(nn)) + " " + str(len(batches[0]))

            nnDif.master_node(nn, batches, labelBatches)
            #err.append(r)

def visitClassicBatches(nn,data,labels, it=1000):
    for c in range(it):
        for cc in range(len(data)):
	  nnS.minibatch_fit(nn, data[cc], labels[cc])

def accuracy(nn, data, label, thr = 0.5):
    predict  = [ np.int8(nnDif.forward(nn,data[c,:]) > thr) == label[c] for c in range(data.shape[0])]
    return 100 * np.double(len(np.where(np.asarray(predict)==False)[0]))/np.double(len(predict))
def accuracyClassic(nn, data, label, thr = 0.5):
    predict  = [ np.int8(nnS.forward(nn,data[c,:]) > thr) == label[c] for c in range(data.shape[0])]
    return 100 * np.double(len(np.where(np.asarray(predict)==False)[0]))/np.double(len(predict))

def group_list(l, group_size):
    for i in xrange(0, len(l), group_size):
        yield l[i:i+group_size]
        
def single_run(te):
    print te
    data, label = make_moons(n_samples=2000, noise=0.05, shuffle=True, random_state = int(time.time()))
        
    data,validation_data,label,validation_label = train_test_split(data,label,train_size = .10)
        #separate the data set into buckets
    
    total_data = []
    for item in data:#list(group_list(data,1))
      total_data.append(np.array(np.matrix(item)))
    
    total_label = []
    for item in label:#list(group_list(label,1))
      
      total_label.append(np.asarray(item))
    #The two separate site sets
    sites_numbers = [2,5,10,15,20,50,60,75,100,150]
    for s in range(len(sites_numbers)):
	nets = []
	nn_groups_data = []
	nn_groups_label = []
	number_of_nets = sites_numbers[s]
	for x in range(number_of_nets):
            nets.append(nnDif.nn_build(1,[2,6,6,1],eta=eta,nonlin=nonlin))
        iters = 700
        group_size = len(total_data) / number_of_nets
        for j in range(number_of_nets):
	    
            x = (total_data[int(float(j)*group_size):int(float((j+1))*group_size)])
            nn_groups_data.append(x)

            nn_groups_label.append(total_label[int(float(j)*group_size):int(float((j+1))*group_size)])
	start = time.time()
	mixed_group_data = []
	mixed_group_label = []
	minim = len(nn_groups_data[0])
	
	for ms in range(len(nn_groups_data)):
	  
	  minim = min(minim,len(nn_groups_data[ms]))
	print "here is size of data: "+str(minim*len(nn_groups_data))
	for se in range(len(nn_groups_data)):
	  for see in range(minim):
	  
	    mixed_group_data.append(np.asarray(nn_groups_data[se][see]))
	    
	    mixed_group_label.append(np.asarray(nn_groups_label[se][see]))
	print len(mixed_group_data)
	minibatched_data, minibatched_label = [x for x in iter_minibatches(minim,mixed_group_data,mixed_group_label)]
	#visitbatches(nets,nn_groups_data,nn_groups_label,[],it=iters)
	visitClassicBatches(nets[0],mixed_group_data,mixed_group_label,it=iters)
	one = accuracy(nets[0], validation_data, validation_label, thr=0.5)

	nn1Acc[te][s] += one
	print "accuracy"
	print one
        '''
        
        number_of_nets = s
        nn_groups_data = []
        nn_groups_label = list()
        groups_data = list()
        groups_label = list()
        nets = list()
        batches = list()
        print total_data
         
        for j in range(number_of_nets):
            x = (total_data[int(float(j)/number_of_nets*(len(total_data))):int(float((j+1))/number_of_nets*(len(total_data)))])
            nn_groups_data.append(x)

            nn_groups_label.append(total_label[int(float(j)/number_of_nets*(len(total_label)/number_of_nets)):int(float((j+1))/number_of_nets*(len(total_label)))])

        nets = list()  #Our differential networks
        batches = [] #a list to store every separate site set
	labelBatches = []

        

            #Build the n site data sets

        for k in range(number_of_nets):
             groups_data.append(np.asarray([item for sublist in nn_groups_data[k] for item in sublist]))        
             groups_label.append(np.asarray([item for sublist in nn_groups_label[k] for item in sublist]))

	#This fills batches, which is a list that contains all of our grouped data
	#The grouped data, in this case, is the data per site
        for ncount in range(number_of_nets):

	  batches.append(groups_data[ncount].T)
	  labelBatches.append(groups_label[ncount])

	err = []
           
	iters = 10000

	#Send the lists of neural nets and batches, which does the same thing as the originial visitbatches
	#but with lists of nets and groups
	
	visitbatches(nets, batches, labelBatches, err, it=iters)
        

	one = accuracy(nets[0], validation_data, validation_label, thr=0.5)

	nn1Acc[te][s/10] += one
	'''
nn1Acc = [[0 for i in range(15)] for j in range(10)]
classAcc1 = [[0 for i in range(15)] for j in range(10)]
classAcc2 = [[0 for i in range(15)] for j in range(10)]
classAcc3 = [[0 for i in range(15)] for j in range(10)]  
number_of_nets = 10

runs = [0,1,2,3,4,5,6]
#single_run(0)
pool.map(single_run,runs)


plt.xlabel("Number of batches [of size 50]")
plt.ylabel("Error rate")
np.savetxt("Many-nn1Acc-datSwitch.txt",nn1Acc)
#np.savetxt("classAcc1-datSwitch.txt",classAcc1)
#np.savetxt("classAcc2-datSwitch.txt",classAcc2)
np.savetxt("Many-classAcc3-datSwitch.txt",classAcc3)
plt.plot(nn1Acc[0], "-r", label = "differential")
plt.plot(nn1Acc[1], "-b", label = "unbksdadsf")
#plt.plot(classAcc2, "-b", label = "classic 1")
#plt.plot(classAcc1, "-g", label = "classic 2")
#plt.plot(classAcc3, "-p", label = "Classic Combined")
plt.legend(loc='upper right')

