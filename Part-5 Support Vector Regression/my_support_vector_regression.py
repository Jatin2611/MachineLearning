# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 19:23:36 2020

@author: Jatin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values

# Splitting the dataset into the Training set and Test set
'''from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) '''

#Feature Scaling (Since SVR doesn't include feature scaling)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

#Fitting regressor model to dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,y)

#Predicting new result 
X_new = [[6.5]]
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_new)))

#Visualizing polynomial linear regression results
y_poly_pred =regressor.predict(X)
plt.scatter(X,y,color="red")
plt.plot(X,y_poly_pred,color="blue")
plt.title("Position-wise Salaries")
plt.xlabel("Position Levels")
plt.ylabel("Salaries")

#Visualizing polynomial linear regression results with high resolution and smoother curve
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color="red")
plt.plot(X_grid,regressor.predict(X_grid),color="blue")
plt.title("Position-wise Salaries")
plt.xlabel("Position Levels")
plt.ylabel("Salaries")
