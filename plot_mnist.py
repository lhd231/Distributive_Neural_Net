# -*- coding: utf-8 -*-
"""
Created on Mon May 16 16:26:09 2016
@author: LHD
"""
import pylab as plt
import numpy as np
import seaborn
import pandas as pd
#plt.xlabel("Number of batches [of size 50]")
#plt.ylabel("Error rate")
arr = [""]*5
arr[0] = np.loadtxt("decent-sameBatch-all-test.txt")
arr[1] = np.loadtxt("cent-sameBatch-all-test.txt")
arr[2] = np.loadtxt("zeros-sameBatch-all-test.txt")
arr[3] = np.loadtxt("eights-sameBatch-all-test.txt")
arr[4] = np.loadtxt("sevens-sameBatch-all-test.txt")
x = range(10,1010,10)
print len(arr[0])
print len(x)
labels = ['decentralized','centralized','site-one','site-two','site-three']

for r,l in zip(arr,labels):
  print "l"
  plt.plot(x,r,label=l)
plt.xlim([10,1000])
plt.xlabel("Number of iterations")
plt.ylabel("accuracy rate")
plt.legend()
plt.show()
'''
decent_plot = plt.plot(decent, label="decentralized")

cent_plot = plt.plot(cent,label="centralized")
print "not cent"
ones_plot = plt.plot(sevens, label="ones")
sevens_plot = plt.plot(eights, label="seven")
zeros_plot = plt.plot(zeros, label="zeros")
plt.legend(handles=[cent_plot])
plt.legend(handles=[cent_plot, decent_plot, ones_plot, sevens_plot, zeros_plot])
plt.show()
'''
#seaborn.factorplot("sample size", hue="class", y="error rate", data=df_long, kind="box")
#seaborn.boxplot(data=dif)
#seaborn.boxplot(data=c3)
#seaborn.boxplot(class2)
#plt.plot(class2)
#plt.errorbar(c2[0],c2[1],c2[2],c2[3],linestyle='None', marker='^')
#plt.plot(classAcc1, "-g", label = "classic 2")
#plt.plot(classAcc3, "-p", label = "Classic Combined")
#plt.legend(loc='upper right')