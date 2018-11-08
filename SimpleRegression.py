# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 15:49:33 2018

@author: Sarthak
"""



from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
R=LinearRegression()
R.fit(X_train,y_train)

y_predicted=R.predict(X_test)

plt.scatter(X_test,y_test,color='green')
plt.plot(X_train,R.predict(X_train),color='blue')


from sklearn.metrics import explained_variance_score
explained_variance_score(y_test,y_predicted)*100