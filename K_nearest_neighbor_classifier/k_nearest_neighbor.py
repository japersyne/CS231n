#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 22:36:00 2019

@author: japersyne
"""

import numpy as np
from past.builtins import xrange

class KNearestNeighbor(object):
    def __init__(self):
        pass
    
    def train(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def compute_distances_two_loops(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in xrange(num_test):
            for j in xrange(num_train):  
                dists[i][j] = np.sqrt(np.sum(np.square(X[i] - self.X_train[j])))
        return dists
    
    def predict_labels(self, dists, k = 1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in xrange(num_test):
            closest_y = []
            closest_y = self.y_train[np.argsort(dists[i][:k])]
            y_pred[i] = np.argmax(np.bincount(closest_y))
            
        return y_pred 
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
        
        
        