#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 14:35:36 2019

@author: japersyne
"""

import random
import numpy as np
from data_utils import load_CIFAR10
import matplotlib.pyplot as plt
from past.builtins import xrange

#加載代碼
cifar10_dir = 'datasets/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

#印出訓練集和測試集大小
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

#取出一些樣本來看看
classes = ['plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 5

for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, 
                            replace = False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()

#取出子集進行練習
num_training = 5000
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

#將圖像數據轉置成二維矩陣
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train.shape, X_test.shape)

#創建knn分類器
from k_nearest_neighbor import KNearestNeighbor

classifier = KNearestNeighbor()
classifier.train(X_train, y_train)

#測試knn分類器
dists = classifier.compute_distances_two_loops(X_test)
print(dists.shape)

#可視化距離矩陣
plt.imshow(dists, interpolation='none')
plt.show()

#k=1(最鄰近演算法)
y_test_pred = classifier.predict_labels(dists, k = 1)

#計算並印出準確率
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print('Nearest Neighbor got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))

#k=5
y_test_pred = classifier.predict_labels(dists, k = 5)

#計算並印出準確率
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print('5 Nearest Neighbor got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))


