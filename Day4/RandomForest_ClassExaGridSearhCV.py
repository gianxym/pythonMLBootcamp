import numpy as np 
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import preprocessing
from sklearn.utils import shuffle
from matplotlib import pyplot
import pickle
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

Signal_file = "../Day3/feature_enginnering/vars_class1.csv"
Bkgd_file = "../Day3/feature_enginnering/vars_class0.csv"

#select the variables you want to use for the algorithm training: 
columns = ["var10","var11","var20","var24","var29","var30","var31","var40","var41","var42","var43","var44","var45","var46",'label'] 

#get data from class 1
sig = pd.read_csv(Signal_file, usecols=columns)
print(sig.head())
print(sig.shape)
#get data from class 0
bkg = pd.read_csv(Bkgd_file, usecols=columns)
#print(bkg.head())
print(bkg.shape)

#Merge signal and Bkgd
data0 = pd.concat([sig,bkg],axis=0)
print(data0.head())

#randomly shuffle the data
data = shuffle(data0, random_state=0) 
#print("after shuffling: ", data.head())
print("data.shape: ", data.shape)

# build train and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:85000,:14], data.iloc[:85000,14], test_size = 0.3, random_state = 100)
print("shape of training dataset: "," x: ",X_train.shape," y: ",y_train.shape)
print("X: ",X_train.head(), " y: ", y_train.head())

tuned_parameters = [{'max_depth': [3,4], 
                     'min_samples_split': [2,3]}]
scores = ['accuracy']
for score in scores:
    
    print()
    print(f"Tuning hyperparameters for {score}")
    print()
    
    clf = GridSearchCV(
        RandomForestClassifier(), tuned_parameters,
        scoring = f'{score}'
    )
    clf.fit(X_train, y_train)
    
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    means = clf.cv_results_["mean_test_score"]
    stds = clf.cv_results_["std_test_score"]
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print(f"{mean:0.3f} (+/-{std*2:0.03f}) for {params}")

