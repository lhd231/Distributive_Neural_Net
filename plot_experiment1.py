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
dif3 = np.subtract(1,np.divide((np.loadtxt("new-3-site-decent-s1-x.txt")[:,:20]),100))
dif2 = np.subtract(1,np.divide((np.loadtxt("new-3-site-decent-s2-x.txt")[:,:20]),100))

dif1 = np.subtract(1,np.divide((np.loadtxt("new-3-site-decent-s3-x.txt")[:,:20]),100))
cent = np.subtract(1,np.divide(np.loadtxt("new-3-site-cent-x.txt")[:,:20],100))
c2 = np.subtract(1,np.divide(np.loadtxt("new-3-site-cent-horn1-x.txt")[:,:20],100))
c1 = np.subtract(1,np.divide(np.loadtxt("new-3-site-cent-horn2-x.txt")[:,:20],100))
c3 = np.subtract(1,np.divide(np.loadtxt("new-3-site-cent-middle-x.txt")[:,:20],100))

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
fw = np.concatenate((dif1,dif2), axis = 0)
fw = np.concatenate((fw,dif3), axis = 0)
fw = np.concatenate((fw,cent), axis = 0)
fw = np.concatenate((fw,c1), axis=0)
fw = np.concatenate((fw,c2), axis=0)
fw = np.concatenate((fw,c3), axis=0)
#fw = np.concatenate((nw, cw), axis=0)
#fw = np.concatenate((fw,c1w), axis=0)
#arrayfire
#fw = np.concatenate((fw,c2w), axis=0)
#fw = np.concatenate((fw,c3w), axis=0)
#fw = c3w
pal = seaborn.color_palette("hls",7)
s = pd.DataFrame(fw, columns=[10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200])
#s = pd.DataFrame({"error rate" : fw, "sample size" : [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360,370,380,390,400,410,420,430,440,450,460,470,480,490,500]})
#exercise = seaborn.load_dataset("exercise")
decent_keys_1 = ["Decentralized site 1"]*6
decent_keys_2 = ["Decentralized site 2"]*6
decent_keys_3 = ["Decentralized site 3"]*6
horn1_keys = ["Horn1"]*6
horn2_keys = ["Horn2"]*6
mid_keys = ["Middle"]*6
cent_keys = ["Centralized"]*6
s["class"] = pd.Series(decent_keys_1 + decent_keys_2 + decent_keys_3 +  cent_keys +horn1_keys + horn2_keys + mid_keys, index = s.index,dtype="category")
df_long = pd.melt(s,"class", var_name="Sample Size", value_name="Accuracy")
#print df_long
#seaborn.boxplot(data = df_long, hue_order="class")
#seaborn.factorplot(data=df_long, x="sample size", hue="class", kind="violin")
#seaborn.factorplot(x="sample size",y="error rate", data=df_long)
pal = seaborn.color_palette("hls",10)
g = seaborn.factorplot(x="Sample Size", y="Accuracy", hue="class",data=df_long,kind="box",palette=pal, size =6, aspect=2, legend = False)
g.despine(left=True)
plt.legend(loc='lower right')
#g.set_ylabels("survival probability")
#seaborn.factorplot(x="sample size", y="error rate",data=df_long, kind="box")
plt.savefig("data_change_3_sites.svg",bbox_inches="tight",pad_inches=0)
seaborn.plt.show(g)
#seaborn.factorplot("sample size", hue="class", y="error rate", data=df_long, kind="box")
#seaborn.boxplot(data=dif)
#seaborn.boxplot(data=c3)
#seaborn.boxplot(class2)
#plt.plot(class2)
#plt.errorbar(c2[0],c2[1],c2[2],c2[3],linestyle='None', marker='^')
#plt.plot(classAcc1, "-g", label = "classic 2")
#plt.plot(classAcc3, "-p", label = "Classic Combined")
#plt.legend(loc='upper right')