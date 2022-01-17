# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 21:14:31 2022

@author: bokyoung
"""

import sys, os
sys.path.append(os.pardir)
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax

# 데이터 불러오기
def getdata():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, one_hot_label=False)
        
    return x_train, t_train
    
# 신경망 구현
def init_network():
    with open('sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)
        
    return network

# 예측함수
def predict(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x,w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,w3) + b3
    y = softmax(a3)
    
    return y

def cross_entropy_error(y,t):
    delta = 1e-7
    if y.ndim==1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
    
    #훈련데이터가 원-핫 벡터라면
    if y.size == t.size:
        t = t.argmax(axis=1)
        
    batch_size = y.shape[0]
    
    return -np.sum(np.log(y[np.arange(batch_size), t]+ delta)) / batch_size


x, t = getdata()
network = init_network()

train_size = x.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x[batch_mask]
t_batch = t[batch_mask]

y_batch = predict(network,x_batch)

print(cross_entropy_error(y_batch, t_batch))


