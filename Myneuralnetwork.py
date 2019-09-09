#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sept 01 17:39:26 2019

@author: amogh
"""

import numpy as np



#sigmoid activation functuon
def sigma(x):
    return 1/(1+np.exp(-x))

#derivative of sigmoid at that value
#refer diary for d/dx (1/1+e^x)
def deriv_sigma(x):
    return x*(1-x)



#dataset 
X = np.array([ [0,1,1], 
               [1,1,1], 
               [0,0,1],
               [1,0,1] ])
 
#labels
y = np.array([[0,1,0,1]]).T
 
#initializing weights, randomly, with mean at 1
#np.random.random initializes values b/w 0 and 1
#neuralnetwork will be a 3x1 matrix
neuralnetwork = 2*np.random.random((3,1)) - 1
 
 
 #iterations
for iter in range(10000):
    x_train = X
    prediction = sigma(np.dot(x_train,neuralnetwork))
    error = y - prediction
    updateval = error * deriv_sigma(prediction)
    neuralnetwork += np.dot(x_train.T,updateval)

print ("Output: ")
print (prediction)
