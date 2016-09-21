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

#c4 = np.loadtxt("up-new-3-site-cent-horn1.txt")
#c3 = np.loadtxt("up-new-3-site-cent-horn2.txt")
#c2 = np.loadtxt("up-new-3-site-cent-middle.txt")
#c1 = np.loadtxt("up-new-3-site-cent.txt")
dif = np.subtract(1,np.divide(np.loadtxt("up-Many-sites-same-samps.txt"),100))
#class2 = np.mean(c2, axis=0)
#print len(c3[0])
#print len(dif[0])
#class3 = np.mean(c3, axis=0)
#class1 = np.mean(c1, axis=0)
#differ = np.mean(dif,axis=0)
#class4 = np.mean(c4, axis=0)
#plt.plot(nn1Acc[0], "-r", label = "differential")
#plt.plot(nn1Acc[1], "-r", label = "differential")
#plt.plot(class2, "-b", label = "classic 1")
#errHigh = class2 + np.std(c2,axis=0)
#error = class3 + np.std(c3,axis=0)
#errLow = differ + np.std(dif,axis=0)
#errup = class1 + np.std(c1,axis=0)
#errDown = class4 + np.std(c4,axis=0)
#x = np.arange(len(class4))
#plt.errorbar(x,class2,yerr=errHigh, label="Classic 1")
#plt.errorbar(x,differ,yerr=errLow, label="differential")
#plt.errorbar(x,class3,yerr=error, label="Classic Combined")
#plt.errorbar(x,class1,yerr=errup, label="Classic 1")
#axe = np.array(['diff' for _ in range(10)])

#print seaborn.load_dataset("tips")
#nw = np.c_[dif,axe]

#axe = np.array(['horn1' for _ in range(10)])
#cw = np.c_[c3,axe]
#axe = np.array(['together' for _ in range(10)])
#c1w = np.c_[c1,axe]
#axe = np.array(['middle' for _ in range(10)])
#c2w = np.c_[c2,axe]
#axe = np.array(['horn2' for _ in range(10)])
#c3w = np.c_[c4,axe]
fw = dif
#fw = np.concatenate((dif,c3), axis = 0)
#fw = np.concatenate((fw,c1), axis=0)
#fw = np.concatenate((fw,c2), axis=0)
#fw = np.concatenate((fw,c4), axis=0)
#fw = np.concatenate((nw, cw), axis=0)
#fw = np.concatenate((fw,c1w), axis=0)
#arrayfire
#fw = np.concatenate((fw,c2w), axis=0)
#fw = np.concatenate((fw,c3w), axis=0)
#fw = c3w
s = pd.DataFrame(fw, columns=[2,4,5,8,10,20,25,40,50,100])
#s = pd.DataFrame({"error rate" : fw, "sample size" : [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360,370,380,390,400,410,420,430,440,450,460,470,480,490,500]})
#exercise = seaborn.load_dataset("exercise")
#decent_keys = ["decent"]*7
#horn1_keys = ["horn1"]*7
#horn2_keys = ["horn2"]*7
#mid_keys = ["mid"]*7
#cent_keys = ["cent"]*7
#s["class"] = pd.Series(decent_keys + horn1_keys + cent_keys + mid_keys + horn2_keys, index = s.index,dtype="category")
df_long = pd.melt(s,var_name="number of sites", value_name="Accuracy")
#print df_long


#seaborn.boxplot(data = df_long, hue_order="class")
#seaborn.factorplot(data=df_long, x="sample size", hue="class", kind="violin")
#seaborn.factorplot(x="sample size",y="error rate", data=df_long)
seaborn.factorplot(x="number of sites", y="Accuracy",data=df_long,kind="box",aspect=2,color="blue")
seaborn.plt.ylim(.6,1)#seaborn.factorplot(x="sample size", y="error rate",data=df_long, kind="box")
seaborn.plt.savefig("moons_variable_sites_same_data.svg",bbox_inches='tight',pad_inches=0)
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