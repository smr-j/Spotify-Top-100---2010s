#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 13:19:33 2023

@author: joschw
"""

from numpy.linalg import linalg
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns; 
#sns.set_theme()
#sns.set_style("darkgrid")
#sns.set(font_scale=2)

# load the data set and convert to numpy array
# this is a data set on colleges
# target variable: Private: Yes or No
origdat = pd.read_csv("data/top_100_cleaned.csv", index_col=[0])
del origdat['top genre']
origdat["top k"] = 1
origdat.loc[origdat["ranking"] > 25, "top k"] = 0
dat = origdat.to_numpy()
# the binary target variable happens to be in
# in the second column of my data set
y = dat[:,1]
# the predictor variables are the remaining
# columns of the data set
Xdf = origdat.iloc[:,6:]
X = dat[:,6:].astype('float')

def logis(x):
    # logistic function = e^x/(1+e^x)
    x = x.astype('float')
    return 1/(1+np.exp(-x))

def logreg(y,X,tol=1e-8):
    # logistic regression function:
        # method:
        # Convex Optimization via Newton's Iteration
        # uses:
        # when the target variable is a binary 0/1 variable
        # conditions:
        # n > p
        # input:
        # y = target variable (n x 1 numpy array)
        # X = design matrix (n x (p+1) numpy array)
        # tol = tolerance (used in stoping crition)
        # output:
        # beta = estimated logistic regression coefficients
    # n = sample size
    # p = number of predictor variables
    n,p = np.shape(X)
    # add the columns of one (for the intercept term)
    # to the design matrix X
    X = np.column_stack((np.ones(n),X))
    Xt = np.transpose(X)
    beta = np.zeros(p+1)    # initialize beta = (0,0,...,0)
    lin = np.matmul(X,beta) # linear predictor
    yhat = logis(lin) # estimated probability of the event y=1
    grad = -np.matmul(Xt,y-yhat) # gradient of logistic loss function
    yhatvar = np.multiply(yhat,1-yhat) # estimated variance of y 
    H = np.matmul(Xt*yhatvar,X) # Hessian matrix of logistic loss function
    beta = beta-np.matmul(np.linalg.inv(H),grad) # update beta vector
    while np.max(np.abs(grad))>tol and np.max(yhatvar)>tol:
        # print (maximum abs value in gradient vector
        # once this value is <tol then the while loop terminates
        print(np.max(np.abs(grad)))
        lin = np.matmul(X,beta) # linear predictor
        yhat = logis(lin) # estimated probability of the event y=1
        grad = -np.matmul(Xt,y-yhat) # gradient of logistic loss function
        yhatvar = np.multiply(yhat,1-yhat) # estimated variance of y 
        H = np.matmul(Xt*yhatvar,X) # Hessian matrix of logistic loss function
        Hinv = np.linalg.inv(H) # inverse of Hessian
        beta = beta-np.matmul(Hinv,grad) # update beta vector
    zscore = np.divide(beta,np.sqrt(np.diag(Hinv))) # z-score of beta estimates
    return beta,zscore



betahat,zscore = logreg(y,X)
# betahat = estimated logistic regression coefficients
# zscore = these are betahat/sqrt(var(betahat)) ~ N(0,1)
#   under the assumption that the true beta = 0

# Hypothesis testing:
# Question: Is variable x_j a useful predictor
#   of y when all other predictor variables
#   x_0, x_1, x_{j-1}, ..., x_{j+1}, ..., , x_{p-1} are
#   included in the model?

# For which variables is the answer to the above
# question equal to "yes"?
Xdf.columns[np.where(np.abs(zscore[1:])>1.96)]
print(Xdf.columns[np.where(np.abs(zscore[1:])>1.96)])

# logistic function plot
linhat = betahat[0]+np.matmul(X,betahat[1:])
plt.plot(linhat,y,'k.',markersize=3)
x = np.linspace(np.floor(np.min(linhat)),np.ceil(np.max(linhat)),100)
plt.plot(x,logis(x))
plt.xlim([-12,12])
plt.title('Plot of 2010-2019 actual ranking versus score')
plt.grid(which='major', axis='x')
plt.show()
plt.close()

## confusion matrix
# Interpretation:
    # conf[0,0] = proportion of obs correctly
    #   classified as y=1 when true value is y=1
    # conf[0,1] = proportion of obs incorrectly
    #   classified as y=1 when true value is y=0
    # conf[1,0] = proportion of obs incorrectly
    #   classified as y=0 when true value is y=1
    # conf[1,1] = proportion of obs correctly
    #   classified as y=0 when true value is y=0
conf = np.zeros([2,2])
conf[0,0] = np.mean(np.logical_and(linhat>0,y==1))
conf[0,1] = np.mean(np.logical_and(linhat>0,y==0))
conf[1,0] = np.mean(np.logical_and(linhat<=0,y==1))
conf[1,1] = np.mean(np.logical_and(linhat<=0,y==0))
print(np.round(conf,3))

k=25
cm = conf
classes = ['Top 25', 'Bottom 75']
title = "Confusion Matrix of 2010-2019 Training Data"
def plot_matrix(cm, classes, title):
  ax = sns.heatmap(cm, cmap="Blues", annot=True, xticklabels=classes, yticklabels=classes, cbar=False)
  ax.set(title=title, xlabel="Actual", ylabel="Predicted")

plot_matrix(cm, classes, title)
plt.show()
plt.close()

conf = np.zeros([2,2])
conf[0,0] = np.sum(np.logical_and(linhat>0,y==1))
conf[0,1] = np.sum(np.logical_and(linhat>0,y==0))
conf[1,0] = np.sum(np.logical_and(linhat<=0,y==1))
conf[1,1] = np.sum(np.logical_and(linhat<=0,y==0))
print(np.round(conf,3))

k=25
cm = conf
classes = ['Top 25', 'Bottom 75']
title = "Confusion Matrix of 2010-2019 Training Data"
def plot_matrix(cm, classes, title):
  ax = sns.heatmap(cm, cmap="Blues", annot=True, fmt='g', xticklabels=classes, yticklabels=classes, cbar=False)
  ax.set(title=title, xlabel="Actual", ylabel="Predicted")

plot_matrix(cm, classes, title)
plt.show()
plt.close()

origdat2 = pd.read_csv("data/2009_top100_cleaned.csv", index_col=[0])
del origdat2['top genre']
origdat2["top k"] = 1
origdat2.loc[origdat2["ranking"] > 25, "top k"] = 0
dat2 = origdat2.to_numpy()
# the binary target variable happens to be in
# in the second column of my data set
y2 = dat2[:,1]
# the predictor variables are the remaining
# columns of the data set
Xdf2 = origdat2.iloc[:,6:]
X2= dat2[:,6:].astype('float')

linhat2 = betahat[0]+np.matmul(X2,betahat[1:])
plt.plot(linhat2,y2,'k.',markersize=3)
x2 = np.linspace(np.floor(np.min(linhat2)),np.ceil(np.max(linhat2)),100)
plt.plot(x2,logis(x2))
plt.xlim([-10,10])
plt.grid(which='major', axis='x')
plt.title('Plot of 2009 actual rankings versus score')
plt.show()
plt.close()

## confusion matrix
# Interpretation:
    # conf[0,0] = proportion of obs correctly
    #   classified as y=1 when true value is y=1
    # conf[0,1] = proportion of obs incorrectly
    #   classified as y=1 when true value is y=0
    # conf[1,0] = proportion of obs incorrectly
    #   classified as y=0 when true value is y=1
    # conf[1,1] = proportion of obs correctly
    #   classified as y=0 when true value is y=0
conf = np.zeros([2,2])
conf[0,0] = np.mean(np.logical_and(linhat2>0,y2==1))
conf[0,1] = np.mean(np.logical_and(linhat2>0,y2==0))
conf[1,0] = np.mean(np.logical_and(linhat2<=0,y2==1))
conf[1,1] = np.mean(np.logical_and(linhat2<=0,y2==0))
print(np.round(conf,3))

k=25
cm = conf
classes = ['Top 25', 'Bottom 75']
title = "Confusion Matrix of 2009 Results"
def plot_matrix(cm, classes, title):
  ax = sns.heatmap(cm, cmap="Blues", annot=True, xticklabels=classes, yticklabels=classes, cbar=False)
  ax.set(title=title, xlabel="Actual", ylabel="Predicted")

plot_matrix(cm, classes, title)
plt.show()
plt.close()

conf = np.zeros([2,2])
conf[0,0] = np.sum(np.logical_and(linhat2>0,y2==1))
conf[0,1] = np.sum(np.logical_and(linhat2>0,y2==0))
conf[1,0] = np.sum(np.logical_and(linhat2<=0,y2==1))
conf[1,1] = np.sum(np.logical_and(linhat2<=0,y2==0))
print(np.round(conf,3))

k=25
cm = conf
classes = ['Top 25', 'Bottom 75']
title = "Confusion Matrix of 2009 Results"
def plot_matrix(cm, classes, title):
  ax = sns.heatmap(cm, cmap="Blues", annot=True, xticklabels=classes, yticklabels=classes, cbar=False)
  ax.set(title=title, xlabel="Actual", ylabel="Predicted")

plot_matrix(cm, classes, title)
plt.show()
plt.close()

origdat3 = pd.read_csv("data/2020_top50_cleaned.csv", index_col=[0])
del origdat3['top genre']
origdat3["top k"] = 1
origdat3.loc[origdat3["ranking"] > 25, "top k"] = 0
dat3 = origdat3.to_numpy()
# the binary target variable happens to be in
# in the second column of my data set
y3 = dat3[:,1]
# the predictor variables are the remaining
# columns of the data set
Xdf3 = origdat3.iloc[:,6:]
X3= dat3[:,6:].astype('float')

linhat3 = betahat[0]+np.matmul(X3,betahat[1:])
plt.plot(linhat3,y3,'k.',markersize=3)
x3 = np.linspace(np.floor(np.min(linhat3)),np.ceil(np.max(linhat3)),50)
plt.plot(x3,logis(x3))
plt.xlim([-8,8])
plt.grid(which='major', axis='x')
plt.title('Plot of 2020 actual rankings versus score')
plt.show()
plt.close()


## confusion matrix
# Interpretation:
    # conf[0,0] = proportion of obs correctly
    #   classified as y=1 when true value is y=1
    # conf[0,1] = proportion of obs incorrectly
    #   classified as y=1 when true value is y=0
    # conf[1,0] = proportion of obs incorrectly
    #   classified as y=0 when true value is y=1
    # conf[1,1] = proportion of obs correctly
    #   classified as y=0 when true value is y=0
conf = np.zeros([2,2])
conf[0,0] = np.mean(np.logical_and(linhat3>0,y3==1))
conf[0,1] = np.mean(np.logical_and(linhat3>0,y3==0))
conf[1,0] = np.mean(np.logical_and(linhat3<=0,y3==1))
conf[1,1] = np.mean(np.logical_and(linhat3<=0,y3==0))
print(np.round(conf,2))

k=25
cm = conf
classes = ['Top 25', 'Bottom 25']
title = "Confusion Matrix of 2020 Results"
def plot_matrix(cm, classes, title):
  ax = sns.heatmap(cm, cmap="Blues", annot=True, xticklabels=classes, yticklabels=classes, cbar=False)
  ax.set(title=title, xlabel="Actual", ylabel="Predicted")

plot_matrix(cm, classes, title)
plt.show()
plt.close()

conf = np.zeros([2,2])
conf[0,0] = np.sum(np.logical_and(linhat3>0,y3==1))
conf[0,1] = np.sum(np.logical_and(linhat3>0,y3==0))
conf[1,0] = np.sum(np.logical_and(linhat3<=0,y3==1))
conf[1,1] = np.sum(np.logical_and(linhat3<=0,y3==0))
print(np.round(conf,2))

k=25
cm = conf
classes = ['Top 25', 'Bottom 25']
title = "Confusion Matrix of 2020 Results"
def plot_matrix(cm, classes, title):
  ax = sns.heatmap(cm, cmap="Blues", annot=True, xticklabels=classes, yticklabels=classes, cbar=False)
  ax.set(title=title, xlabel="Actual", ylabel="Predicted")

plot_matrix(cm, classes, title)
plt.show()
plt.close()
