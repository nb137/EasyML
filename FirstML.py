#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 16:11:01 2017

This isn't actually my first ML

@author: nbrunner
"""

import pandas as pd
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Presume that we are in working path, as should be in this folder
filename = "iris.data"
cols = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'] # It would be great if this was a part of the file
dataset = pd.read_csv(filename, names=cols)

#print("Shape:")
#print(dataset.shape)

# If you are working in console or notebook
#dataset.describe

#print(dataset.groupby('class').size())

#dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#plt.show()

#dataset.hist()
#plt.show()

#scatter_matrix(dataset)

'''
Separate into Train and Test datasets

Webpage uses array as a variable name but this is a bad choice bc it overwrites a global
'''

a = dataset.values
x = a[:,0:4]    # x values use all columns except flower name
y = a[:,4]      # y vals are flower name (to be fit)

validation_size = 0.2
# seed = 7  # Comment out because I don't need to match webpage

x_t, x_v, y_t, y_v = model_selection.train_test_split(x, y, test_size=validation_size)

# for the sake of writing a script I think we should create the list instead of appending everything
models = [('LR', LogisticRegression()), 
        ('KNN', KNeighborsClassifier()), ('CART', DecisionTreeClassifier()),
        ('NB', GaussianNB()), ('SVM', SVC())]

# removed ('LDA', LinearDiscriminantAnalysis()), because kernel kept dying?

results = []
names = []
seed=7  # Use this so random selections in the for loop select same data

for name, model  in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, x_t, y_t, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" %(name, cv_results.mean(), cv_results.std())
    print(msg)
    
# Run KNN predictions
knn = KNeighborsClassifier()
knn.fit(x_t, y_t)
predictions = knn.predict(x_v)
print(accuracy_score(y_v, predictions))
print(confusion_matrix(y_v, predictions))
print(classification_report(y_v, predictions))
