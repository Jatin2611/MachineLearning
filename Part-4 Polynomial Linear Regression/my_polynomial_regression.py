# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 19:39:47 2020

@author: Jatin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
'''from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) '''

#Fitting dataset to linear Regression
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X,y)

#Fitting dataset to polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_regressor = PolynomialFeatures(degree = 4)
X_poly = poly_regressor.fit_transform(X)
#Fit x_poly with multiple columns into multiple regressor
multi_regressor = LinearRegression()
multi_regressor.fit(X_poly, y)

#Visualising Linear regression results
y_linear_pred = linear_regressor.predict(X)
plt.scatter(X,y,color="red")
plt.plot(X,y_linear_pred,color="blue")
plt.title("Position-wise Salaries")
plt.xlabel("Position Levels")
plt.ylabel("Salaries")

#Visualizing polynomial linear regression results
y_poly_pred = multi_regressor.predict(poly_regressor.fit_transform(X))
plt.scatter(X,y,color="red")
plt.plot(X,y_poly_pred,color="blue")
plt.title("Position-wise Salaries")
plt.xlabel("Position Levels")
plt.ylabel("Salaries")

#Predicting new result with linear regression
X_new = [[6.5]]
linear_regressor.predict(X_new)
#Predicting new result with polynomial regression
multi_regressor.predict(poly_regressor.fit_transform(X_new))