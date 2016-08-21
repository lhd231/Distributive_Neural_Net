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

#c4 = np.loadtxt("classic_combined_20k_wgtSample.txt")
#c3 = np.loadtxt("classic1_20k_wgtSample.txt")


#HERE


#c2 = np.loadtxt("classAcc-lowiters-gaussProb.txt")
#c1 = np.loadtxt("classAcc1-lowiters-gaussProb.txt")
dif = np.loadtxt("classAcc-lowiters-gaussProb.txt")
#dif = np.loadtxt("nn1Acc-lowiters-gaussProb.txt")
dif = np.delete(dif,-1,0)
dif = np.delete(dif,-1,0)
dif = np.delete(dif,-1,0)
dif = np.delete(dif,0,1)
#dif = np.loadtxt("Many-nn1Acc-datSwitch.txt")

#class2 = np.mean(c2, axis=0)

#class3 = np.mean(c3, axis=0)
#class1 = np.mean(c1, axis=0)
differ = np.mean(dif,axis=0)
#class4 = np.mean(c4, axis=0)
#plt.plot(nn1Acc[0], "-r", label = "differential")
#plt.plot(nn1Acc[1], "-r", label = "differential")
#plt.plot(class2, "-b", label = "classic 1")
#errHigh = class2 + np.std(c2,axis=0)
#error = class3 + np.std(c3,axis=0)
errLow =  np.std(dif,axis=0)
#errClass = np.std(c1,axis=0)
#errFull = np.std(c2,axis=0)
#print np.std(dif,axis=0)
#print differ
#print differ + np.std(dif,axis=0)
#print errLow
#errup = class1 + np.std(c1,axis=0)
#errDown = class4 + np.std(c4,axis=0)
#x = np.arange(len(class4))
#plt.errorbar(x,class2,yerr=errHigh, label="Classic 1")
#plt.errorbar(x,differ,yerr=errLow, label="differential")
#plt.errorbar(x,class3,yerr=error, label="Classic Combined")
#plt.errorbar(x,class1,yerr=errup, label="Classic 1")
#axe = np.array(["diff" for _ in range(7)])

#print seaborn.load_dataset("tips")
#nw = np.c_[dif,axe]

#axe = np.array(["c3" for _ in range(10)])
#cw = np.c_[c3,axe]
#axe = np.array(["c1" for _ in range(10)])
#c1w = np.c_[c1,axe]
#axe = np.array(["c2" for _ in range(10)])
#c2w = np.c_[c2,axe]
#axe = np.array(["cc" for _ in range(10)])
#c3w = np.c_[c4,axe]
#fw = np.concatenate((nw, cw), axis=0)
#fw = np.concatenate((fw,c1w), axis=0)
#arrayfire
#fw = np.concatenate((fw,c2w), axis=0)
#fw = np.concatenate((fw,c3w), axis=0)
#fw = c3w
fw = dif
#s = pd.DataFrame({'y':differ, 'x':[10,20,30,40,50,60,70,80,90,100,110,120,130,140]})
x = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360,370,380,390,400,410,420,430,440,450,460,470,480,490]
#s.plot('x','y', legend=False)
seaborn.boxplot(data = fw, hue_order="class")
#print s
#df_long = pd.melt(s, "class", var_name="sample size", value_name="error rate")
#print df_long
#plt.ylim([0,100])
#plt.ylabel("Error rate")
#plt.xlabel("Group size (3 groups per decentralized line")
#plt.xticks( differ, [20,40,60,80,100,120,140,160,180,200,220,240,260,280] )
#plt.xlim([10,160])
#plt.plot(s)
#print differ.shape[0]
#plt.errorbar(x, differ, yerr = errLow)
#plt.errorbar(x, class1, yerr = errClass)
#plt.errorbar(x, class2, yerr = errFull)
#seaborn.factorplot(data=s, kind="box")
#seaborn.factorplot(x="sample size", y="error rate", hue="class",data=df_long, kind="box")
seaborn.plt.show()
#seaborn.factorplot("sample size", hue="class", y="error rate", data=df_long, kind="box")
#seaborn.boxplot(data=dif)
#seaborn.boxplot(data=c3)
#seaborn.boxplot(class2)
#plt.plot(class2)
#plt.errorbar(c2[0],c2[1],c2[2],c2[3],linestyle='None', marker='^')
#plt.plot(classAcc1, "-g", label = "classic 2")
#plt.plot(classAcc3, "-p", label = "Classic Combined")
#plt.legend(loc='upper right')
