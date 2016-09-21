# -*- coding: utf-8 -*-
"""
Created on Mon May 16 16:26:09 2016
@author: LHD
"""
import pylab as plt
import numpy as np
import seaborn
import pandas as pd
import collections

rand_mat =np.random.choice([0,1],size=(20,784),p=[3./4,1./4])
unique, counts = np.unique(rand_mat, return_counts=True)
print dict(zip(unique,counts))
#plt.xlabel("Number of batches [of size 50]")
#plt.ylabel("Error rate")
#print len(np.loadtxt("cent-sameBatch-all-2.txt")[0])
#c4 = np.loadtxt("decent-remGrad-all-s2.txt")[:,:20]#np.loadtxt("zeros-sameBatch-all-test-2.txt")[:,:20]
c3 = np.loadtxt("cent-remGrad-noFancy.txt")[:,:30][:,1::3]#[:,1::50]#np.loadtxt("cent-sameBatch-all-test-5.txt")[:,:50]#[:,:10]
c2 = np.loadtxt("decent-remGrad-s1-noFancy.txt")[:,:30][:,1::3]#[:,1::50]#np.loadtxt("eights-sameBatch-all-test-5.txt")[:,:50]#[:,:10]
c1 = np.loadtxt("decent-remGrad-s2-noFancy.txt")[:,:30][:,1::3]#[:,1::50]#np.loadtxt("sevens-sameBatch-all-test-5.txt")[:,:50]#[:,:10]
dif = np.loadtxt("decent-remGrad-s3-noFancy.txt")[:,:30][:,1::3]#[:,1::50]#np.loadtxt("decent-sameBatch-all-test-5.txt")[:,:50]#[:,:10]
cSite1 = np.loadtxt("eights-remGrad-noFancy.txt")[:,:30][:,1::3]#[:,1::50]
cSite2 = np.loadtxt("sevens-remGrad-noFancy.txt")[:,:30][:,1::3]#[:,1::50]
cSite3 = np.loadtxt("zeros-remGrad-noFancy.txt")[:,:30][:,1::3]#[:,1::50]


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
pal = seaborn.color_palette("hls",7)
fw = np.concatenate((c3,c2), axis = 0)
fw = np.concatenate((fw,c1), axis=0)
fw = np.concatenate((fw,dif), axis=0)
fw = np.concatenate((fw,cSite1),axis=0)
fw = np.concatenate((fw,cSite2),axis=0)
fw = np.concatenate((fw,cSite3),axis=0)
#fw = np.concatenate((fw,c4), axis=0)
#fw = np.concatenate((nw, cw), axis=0)
#fw = np.concatenate((fw,c1w), axis=0)
#arrayfire
#fw = np.concatenate((fw,c2w), axis=0)
#fw = np.concatenate((fw,c3w), axis=0)
#fw = c3w
col=range(30,330,30)
s = pd.DataFrame(fw, columns=col)
#s = pd.DataFrame({"error rate" : fw, "sample size" : [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360,370,380,390,400,410,420,430,440,450,460,470,480,490,500]})
#exercise = seaborn.load_dataset("exercise")
decent_keys = ["Decentralized site1"]*6
#horn1_keys = ["site1"]*6
horn2_keys = ["Decentralized site2"]*6
mid_keys = ["Decentralized site3"]*6
cent_keys = ["Centralized Pool"]*6
site1_keys = ["Centralized site1"]*6
site2_keys = ["Centralized site2"]*6
site3_keys = ["Centralized site3"]*6
s["class"] = pd.Series(cent_keys + decent_keys + horn2_keys + mid_keys + site1_keys + site2_keys + site3_keys, index = s.index,dtype="category")
df_long = pd.melt(s,"class", var_name="Number of Iterations", value_name="Accuracy")
#print df_long
#seaborn.boxplot(data = df_long, hue_order="class")
#seaborn.factorplot(data=df_long, x="sample size", hue="class", kind="violin")
#seaborn.factorplot(x="sample size",y="error rate", data=df_long)

g = seaborn.factorplot(x="Number of Iterations", y="Accuracy", hue="class",data=df_long,kind="box", palette = pal,linewidth=.8,size=5, aspect=2, legend = False)
g.despine(left=True)
plt.legend(loc='lower right')
#g.set_ylabels("survival probability")

#seaborn.factorplot(x="sample size", y="error rate",data=df_long, kind="box")
seaborn.plt.savefig("removed_gradients.svg",bbox_inches="tight",pad_inches=0)
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