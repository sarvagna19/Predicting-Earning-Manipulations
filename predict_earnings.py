# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 15:53:50 2018

@author: Mayurakshi
"""

# Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('C:/Users/Mayurakshi/Desktop/mba/2nd semester/predictive analytics/predicting_earnings/Earnings_model.csv')
X = dataset.iloc[:,1:10].values
y = dataset.iloc[:, 10].values

#Encoding the categorical data
from sklearn.preprocessing import LabelEncoder
labelEncoder_X=LabelEncoder()
X[:,8]=labelEncoder_X.fit_transform(X[:,8])


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)



# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))



# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

