# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 17:57:06 2018

@author: Sarthak
"""

import numpy as  np
import matplotlib.pyplot as plt
import pandas as pd


Dataset=pd.read_csv("50_Startups.csv")
X=Dataset.iloc[:,:-1].values
y=Dataset.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()
X[:, 3]=labelencoder.fit_transform(X[:,3])
onehotencoder=OneHotEncoder()
X = onehotencoder.fit_transform(X).toarray()

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
R=LinearRegression()
R.fit(X_train,y_train)


y_pred=R.predict(X_test)
