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

def gauss(x,mu,sigma):
  return norm(mu,[[sigma,0],[0, sigma]]).pdf(x)


def bias(data, label, bias):
  dnew = []
  lnew = []
  count = 0
  totalCount = 500
  for x,y in zip(data,label):
    if totalCount > 0:
      if y == 0 and count < totalCount *bias:
	dnew.append(x)
	lnew.append(y)
	count += 1
	totalCount -= 1
      elif y == 1:
	dnew.append(x)
	lnew.append(y)
	totalCount -= 1
  return dnew, lnew


#adagrad and adadelta  esp adadelta
def iter_minibatches(chunksize, data):
    # Provide chunks one by one
    chunkstartmarker = 0
    numsamples = data.shape[1]
    while chunkstartmarker < numsamples:
        chunkrows = range(chunkstartmarker,chunkstartmarker+chunksize)
        X_chunk = data[:,chunkrows]
        yield X_chunk
        chunkstartmarker += chunksize


def visitbatches(nn, batches, labels, errlist, it=1000):
    for c in range(it):
      nnDif.master_node(nn,batches,labels)
        #for i in range(len(batches)):
         #   batch = batches[i]
          #  cc = np.mod(c,len(batch))
           # nnDif.master_node(nn, batch[cc][0], batch[cc][1])
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
        yield np.asarray(l[i:i+group_size]).T
#TODO:  make these guys 2D.  So we can avoid the problem of [(i+j)/2  + (k +l+m+n)] / 4

def single_run(s):
    print s
    siteCount = 0
    if s == 2:
      siteCount = 0
    if s == 10:
      siteCount = 1
    if s == 100:
      siteCount = 2
    
        #separate the data set into buckets
    #TODO: Move this guy to the loop
    #TODO: Split data and label

    
    #The two separate site sets
    nat = []
    #TODO:  Set this guy to an array. 2, 10, 100
    
    for te in range(10):
      data, label = make_moons(n_samples=2000, noise=0.05, shuffle=True, random_state = int(time.time()))
        
      data,validation_data,label,validation_label = train_test_split(data,label,train_size = .60)
      print te
      biases = [.02,.04,.08,.14,.20,.25,.30,.35,.40,.45,.50]
      biasCount = 0
      for b in biases:

	nData, nLabel = (bias(data,label,b))

	total_data = list(group_list(nData,1))
	total_label = list(group_list(nLabel,1))
        number_of_nets = s

        nn_groups_data = []
        nn_groups_label = list()
        groups_data = list()
        groups_label = list()
        nets = list()
        batches = list()
        for j in range(number_of_nets):
	    dataCount = len(total_data) / s
            #x = (total_data[int(float(j)/number_of_nets*(len(total_data))):int(float((j+1))/number_of_nets*(len(total_data)))])
            x = total_data[dataCount * j : dataCount*(j+1)]

            nn_groups_data.append(x)
            #print len(nn_groups_data[j])
            #print "HERE BREAK"
            #print str(float((j+1))) + "  " + str(number_of_nets) + "  LEN TOTAL DATA: " + str(len(total_data)) 
            #print int(float((j+1))/number_of_nets*(len(total_data)/number_of_nets))
            #print "HERE END"
            nn_groups_label.append(total_label[dataCount * j : dataCount*(j+1)])
        #print len(nn_groups_data[1])
        print "HERE IS DATA LENGTH  " + str(len(nn_groups_data[0]))
        print "HERE ARE NNS NUMB    " + str(len(nn_groups_data))
        nets = list()  #Our differential networks
        batches = list() #a list to store every separate site set
        labelBatches = list()
        #Lists for our error to be plotted later
        #FIXME:  Needs to fill batches only.  Really though?  Think about this more...
        #TODO:  run this guy when you get the chance.  10 samples, and check.  if it's not what you want,
        #it's a problem...        
        #for i in range(len(nn_groups_data)):
            
            #nnClassic1 = nnS.nn_build(1,[2,6,6,1],eta=eta,nonlin=nonlin)

        
        for x in range(number_of_nets):
            nets.append(nnDif.nn_build(1,[2,6,6,1],eta=eta,nonlin=nonlin))


        for k in range(number_of_nets):
             groups_data.append(np.asarray([item for sublist in nn_groups_data[k] for item in sublist]))        
             groups_label.append(np.asarray([item for sublist in nn_groups_label[k] for item in sublist]))
             total_groups_data = np.asarray([item for sublist in groups_data for item in sublist])
             total_groups_label =  np.asarray([item for sublist in groups_label for item in sublist])
             #Our classic combined nn    
             nnClassic3 = nnS.nn_build(1,[2,6,6,1],eta=eta,nonlin=nonlin)
	    
        for ncount in range(number_of_nets):
	  batches.append(groups_data[ncount])
	  labelBatches.append(groups_label[ncount])
            #t = [x for x in iter_minibatches(2,total_groups_data.T, total_groups_label)]
            #t3 = [x for x in iter_minibatches(2,total_groups_data.T, total_groups_label)]
	err = []
            #Run the batches through the algos
	iters = 10000
            #visitClassicBatches(nnClassic1,t, it=iters)
            #visitClassicBatches(nnClassic3,t3, it=iters)
	start = time.time()
	visitbatches(nets, nn_groups_data, nn_groups_label, err, it=iters)
        print time.time() - start
            #calculate error
            #classic = accuracyClassic(nnClassic1,validation_data,validation_label, thr=0.5)
	one = accuracy(nets[0], validation_data, validation_label, thr=0.5)
            #classic3 = accuracyClassic(nnClassic3,validation_data,validation_label, thr=0.5)
	nat = nets[0]
            #build plottable arrays
	nn1Acc[siteCount][te][biasCount] += one
	print one
            #classAcc1[te][s/2] += classic
            #classAcc3[te][s/2] += classic3

            #print "us " + str(one) + " c1 " + str(classic) + " cc " + str(classic3)

	biasCount +=1
    siteCount += 1
nn1Acc = [[[0 for k in range(11)] for i in range(10)] for j in range(3)]

number_of_nets = 10
sites = [2,10,100]
siteCount = [0,1,2]
single_run(100)
#pool.map(single_run,sites)
    
	
        #classAcc3[te][s/2] /= len(nn_groups_data)
#nn1Acc[:] = [x / 17 for x in nn1Acc]
#classAcc1[:] = [x / 17 for x in classAcc1]
#classAcc3[:] = [x / 17 for x in classAcc3]
plt.xlabel("Number of batches [of size 50]")
plt.ylabel("Error rate")
np.savetxt("diff_biases-2sites.txt",nn1Acc[0])
np.savetxt("diff_biases-100sites.txt",nn1Acc[1])
np.savetxt("diff_biases-500sites.txt",nn1Acc[2])
#np.savetxt("classAcc1-datSwitch.txt",classAcc1)
#np.savetxt("classAcc2-datSwitch.txt",classAcc2)
#np.savetxt("Many-classAcc3-datSwitch.txt",classAcc3)
#plt.plot(nn1Acc[0], "-r", label = "differential")
#plt.plot(nn1Acc[1], "-b", label = "unbksdadsf")
#plt.plot(classAcc2, "-b", label = "classic 1")
#plt.plot(classAcc1, "-g", label = "classic 2")
#plt.plot(classAcc3, "-p", label = "Classic Combined")
plt.legend(loc='upper right')

