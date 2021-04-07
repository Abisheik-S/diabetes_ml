# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from genetic_selection import GeneticSelectionCV

diabetes = pd.read_csv('D:/abi diabetes/diabetes.csv')
print(diabetes.columns)
diabates1=diabetes[(diabetes.BloodPressure != 0) & (diabetes.BMI != 0) & (diabetes.Glucose != 0)]

from sklearn.decomposition import PCA
pca = PCA(n_components=8)
principalComponents = pca.fit_transform(diabates1.loc[:, diabates1.columns != 'Outcome'])
import seaborn as sns
#sns.countplot(diabetes['Outcome'],label="Count")
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(diabates1.loc[:, diabates1.columns != 'Outcome'], diabates1['Outcome'],test_size=0.25, random_state=1234)
X=principalComponents
Y=diabates1["Outcome"]

#%%

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVC', SVC()))
models.append(('LR', LogisticRegression()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('GNB', GaussianNB()))
models.append(('RF',RandomForestClassifier(max_depth=5,random_state=1234 )))
models.append(('GB', GradientBoostingClassifier()))

names = []
trscores = []
tescore=[]
for name, model in models:
    clf=model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    trscores.append(clf.score(X_train, y_train))
    names.append(name)
    tescore.append(clf.score(X_test, y_test))
tr_split = pd.DataFrame({'Name': names, 'Training': trscores,'Testing':tescore})
print(tr_split)


#%%
from sklearn.model_selection import cross_val_score,KFold
names = []
scores = []
for name, model in models:
    
    kfold = KFold(n_splits=5, random_state=1234) 
    score = cross_val_score(model, X, Y, cv=kfold, scoring='accuracy').mean()
    
    names.append(name)
    scores.append(score)
kf_cross_val = pd.DataFrame({'Name': names, 'Score': scores})
print(kf_cross_val)
#%%
clf = RandomForestClassifier( max_depth=5,random_state=1234)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

print("Accuracy on training set: {:.3f}".format(clf.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(clf.score(X_test, y_test)))

#%%
import pickle
pickle.dump(clf,open('diamodel.pkl','wb'))
#%%
cl1f=pickle.load(open('diamodel.pkl','rb'))

#%%
a=pd.DataFrame({'Pregnancies':1, 'Glucose':2, 'BloodPressure':3, 'SkinThickness':9, 'Insulin':0,
       'BMI':7, 'DiabetesPedigreeFunction':8, 'Age':9},index=[0])
res=clf.predict(a)
print(res)
