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

decent = np.loadtxt("decent1.txt")
cent = np.loadtxt("cent1.txt")
zeros = np.loadtxt("zeros1.txt")
eights = np.loadtxt("eights1.txt")
sevens = np.loadtxt("sevens1.txt")

plt.plot(decent)
plt.plot(sevens)
plt.plot(cent)
#seaborn.factorplot("sample size", hue="class", y="error rate", data=df_long, kind="box")
#seaborn.boxplot(data=dif)
#seaborn.boxplot(data=c3)
#seaborn.boxplot(class2)
#plt.plot(class2)
#plt.errorbar(c2[0],c2[1],c2[2],c2[3],linestyle='None', marker='^')
#plt.plot(classAcc1, "-g", label = "classic 2")
#plt.plot(classAcc3, "-p", label = "Classic Combined")
#plt.legend(loc='upper right')