import pandas as pd
import numpy  as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

#load data
Signal_file = "vars_class1.csv"
Bkgd_file = "vars_class0.csv"

#get signal data
sig = pd.read_csv(Signal_file) #, usecols=columns)
#print(sig.head())
print(sig.shape)
#get Bkgd data
bkg = pd.read_csv(Bkgd_file) #, usecols=columns)
#print(bkg.head())
print(bkg.shape)

#Merge signal and Bkgd
data0 = pd.concat([sig,bkg],axis=0)
#print(data0.head())

#randomly shuffle the data
data = shuffle(data0, random_state=0) 
print("after shuffling: ", data.head())
print("data.shape: ", data.shape)

'''
plt.figure(figsize=(15,7))
plt.hist(sig["var10"], bins=100, label='var1-class1', alpha=0.5, color='r',log=True)
plt.hist(bkg["var10"], bins=100, label='var1-class0', alpha=0.5, color='b',log=True)
plt.xlabel('Var10', fontsize=25)
plt.ylabel('Number of entries', fontsize=25)
plt.legend(fontsize=15)
plt.tick_params(axis='both', labelsize=25, pad=5)
plt.show() 
'''

def make_graphs(input_list):
    list=input_list
    i=0
#    for i in range(46):
    for i in range(9,10):
        x1=list.loc[list.label==0]
        x2=list.loc[list.label==1]
        n,bins,patches =plt.hist(x1.iloc[:,i],50,density=1,facecolor='r',alpha=0.45)
        n,bins,patches =plt.hist(x2.iloc[:,i],50,density=1,facecolor='g',alpha=0.45)
        print("Variable:",i+1)
        plt.savefig('Variable'+str(i+1)+'.png')
        plt.close()
        i=i+1

make_graphs(data)
