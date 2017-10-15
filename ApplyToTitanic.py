#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 10:38:43 2017

Adapt FirstML to the titanic dataset

We will need to work a lot more on data cleaning for this
Ideas from
https://rstudio-pubs-static.s3.amazonaws.com/98715_fcd035c75a9b431a84efca8b091a185f.html


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

fn1 = "titanictrain.csv"
fn2 = "titanictest.csv"

# unlike First, the column headers are included in loading
# And we also need to clean up NaNs

ds1 = pd.read_csv(fn1)

# Count NaN values per column to help decide later what areas of data we may want to ignore
cols = list(ds1.columns)
nancheck = []
for i in cols:
    nancheck.append((i,ds1[i].isnull().sum()))
# You will see age is the most missing

# Source mentioned looking at class to infer age
ds1.boxplot(column="Age", by="Pclass")
ds1.boxplot(column="Age", by="SibSp")   # num siblings is also an indicator

# We could get lazy and just use average ages here
# Or we can ignore the missing age columns (177/891 = 20%)
# Or we can learn from the data and do a whole thing

# Training set is dataset where there are no NaNs
#x_t = ds1[ds1["Age"].notnull()][["Pclass","SibSp"]]
x_t = ds1[ds1[["Age","SibSp"]].notnull()][["SibSp"]]
y_t = ds1[ds1[["Age","SibSp"]].notnull()]["Age"]
# Predict set is ds where Age is nan
x_p = ds1[ds1["Age"].isnull()][["Pclass","SibSp"]]
# y_p will be placed into null values

''' Use ML from FirstML Code'''
models = [('LR', LogisticRegression()), 
        ('KNN', KNeighborsClassifier()), ('CART', DecisionTreeClassifier()),
        ('NB', GaussianNB()), ('SVM', SVC())]
results = []
names = []
seed = 7
for name, model  in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, x_t, y_t, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" %(name, cv_results.mean(), cv_results.std())
    print(msg)

''' Create data cleaning function for the Titanic Dataset'''
def tit_clean(dataset):
    # Cabin - some data cleaning strategies ignore this
    # I will set nan=0 and then use the first letter of the cabin otherwise
    dataset["Cabin"].fillna(0,inplace=True)
    dataset["Cabin"] = dataset["Cabin"].astype(str).str[0]
