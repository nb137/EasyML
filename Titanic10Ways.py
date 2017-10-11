# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 16:35:19 2017

Based off of http://benalexkeen.com/decision-tree-classifier-in-python-using-scikit-learn/
and https://github.com/savarin/pyconuk-introtutorial
@author: Nathan.Brunner
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

dfall = pd.read_csv("titanic.csv", index_col = "PassengerId")

df = dfall[["Pclass", 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']]
'''
Rather than drop NA we should come back and use other techniques to "guess" NA values
'''

df = df.dropna()    # From 891 records to 714
df["Sex"] = df["Sex"].map({'male':0, 'female':1})


X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

model = tree.DecisionTreeClassifier()

model.fit(X_train, y_train)

y_predict = model.predict(X_test)
accuracy_score(y_test, y_predict) # 0.83 from benalexkeen, 0.82 from my first run
confusion_matrix(y_test, y_predict)


'''
Start PyConUk tutorial influenced part
'''
from sklearn.ensemble import RandomForestClassifier

df2 = dfall.drop(["Name", "Ticket", "Cabin"], axis=1)   # Savarin version of Dataset
df2 = df2.dropna()
df2["Sex"] = df2["Sex"].map({'male':0, 'female':1})
df2['Embarked'] = df2['Embarked'].map({'C':1, 'S':2, 'Q':3}).astype(int)

model2 = RandomForestClassifier(n_estimators=100)
X2 = df2.drop("Survived", axis=1)
y2 = df2["Survived"]
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, random_state=1)

model2 = model2.fit(X_train2, y_train2)
y_predict2 = model2.predict(X_test2)
accuracy_score(y_test2, y_predict2) 
# 0.775 on first run w/ nan dropped


''' Re-do with mode, mean instead of Drop NA'''
from scipy.stats import mode

df3 = dfall.drop(["Name", "Ticket", "Cabin"], axis=1)
df3["Age"] = df3["Age"].fillna(df["Age"].mean())
df3["Sex"] = df3["Sex"].map({'male':0, 'female':1})
df3["Embarked"] = df3["Embarked"].fillna(mode(df3["Embarked"].dropna())[0][0])
df3['Embarked'] = df3['Embarked'].map({'C':1, 'S':2, 'Q':3}).astype(int)

model3 = RandomForestClassifier(n_estimators=100)
X3 = df3.drop("Survived", axis=1)
y3 = df3["Survived"]
X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, random_state=1)

model3 = model3.fit(X_train3, y_train3)
y_predict3 = model3.predict(X_test3)
accuracy_score(y_test3, y_predict3) 
# 0.776 on second run w/ median embarked value

''' Re-do with dummy variables'''
df4 = df3
# Dummy variables means changing the embarked locations to binary 3 columns
df4 = pd.concat([df4, pd.get_dummies(dfall["Embarked"], prefix="Embarked")], axis=1)
df4 = df4.drop(["Embarked"], axis=1)

model4 = RandomForestClassifier(n_estimators=100)
X4 = df4.drop("Survived", axis=1)
y4 = df4["Survived"]
X_train4, X_test4, y_train4, y_test4 = train_test_split(X4, y4, random_state=1)

model4 = model4.fit(X_train4, y_train4)
y_predict4 = model4.predict(X_test4)
accuracy_score(y_test4, y_predict4) 
# 0.776 on third run w/ dummy variables

model5 = RandomForestClassifier(n_estimators = 100, max_features=0.5, max_depth=5.0)
# Features and depth tuned with sklearn.grid_search
model5.fit(X_train4, y_train4)
y_predict5 = model5.predict(X_test4)
accuracy_score(y_test4, y_predict5)
# 0.785 on fourth run, change parameters

''' Support Vector Classifier'''
from sklearn.svm import SVC
model6 = SVC(kernel='linear')
X6 = df4.drop("Survived", axis=1)
y6 = df4["Survived"]
X_train6, X_test6, y_train6, y_test6 = train_test_split(X6, y6, random_state=1)
model6.fit(X_train6, y_train6)
y_predict6 = model6.predict(X_test6)
accuracy_score(y_test6, y_predict6)
#0.785 is something weird happening with similar values coming out?

''' SVC with grid search'''
from sklearn.grid_search import GridSearchCV
parameter_grid = {
        'C': [1, 10],
        'gamma': [0.1, 1]}
grid_search = GridSearchCV(SVC(kernel='linear'), parameter_grid, cv=5, verbose=3)
grid_search.fit(X_train6, y_train6)
sorted(grid_search.grid_scores_, key=lambda x:x.mean_validation_score)
grid_search.best_score_
# Use C = 1, gamma = 0.1
model7 = SVC(kernel='linear', C=1, gamma=0.1)
model7.fit(X_train6, y_train6)
y_predict7 = model7.predict(X_test6)
accuracy_score(y_test6, y_predict7)
# Still 0.785....