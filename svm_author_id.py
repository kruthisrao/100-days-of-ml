#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

clf=SVC(kernel="rbf",C=10000.0)

t0 = time()
clf.fit(features_train,labels_train)
print "Time to train: ", round((time() - t0), 3), "s"
t1 = time()
pred = clf.predict(features_test)
print(pred)
print "Time to predict: ", round((time() - t1), 3), "s"
acc=accuracy_score(pred,labels_test)
print(acc)
a=pred[10]
b=pred[26]
c=pred[50]
print(a)
print(b)
print(c)
i=0;count=0
for i in range(len(pred)):
	if pred[i]==1:
		count=count+1
print(count)
#########################################################


