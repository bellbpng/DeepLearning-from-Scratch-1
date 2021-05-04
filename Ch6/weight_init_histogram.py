# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLU(x):
    return np.maximum(0, x)


def tanh(x):
    return np.tanh(x)
    
input_data = np.random.randn(1000, 100)  # 1000개의 데이터
node_num = 100  # 각 은닉층의 노드(뉴런) 수
hidden_layer_size = 5  # 은닉층이 5개
# 이곳에 활성화 결과를 저장
activations1 = {}  
activations2 = {}
activations3 = {}
activations4 = {}

x = input_data

for i in range(hidden_layer_size):
    if i != 0:
        x = activations1[i-1]       
        x = activations2[i-1]        
        x = activations3[i-1]
        x = activations4[i-1]

    # 초깃값을 다양하게 바꿔가며 실험해보자！
    # 표준편차가 1인 정규분포
    w1 = np.random.randn(node_num, node_num) * 1

    # 표준편차가 0.01인 정규분포 
    w2 = np.random.randn(node_num, node_num) * 0.01

    # Xavier 초깃값 
    w3 = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num) 

    # He 초깃값(ReLU에 특화된 초기값)
    w4 = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)


    a1 = np.dot(x, w1)
    a2 = np.dot(x, w2)
    a3 = np.dot(x, w3)
    a4 = np.dot(x, w4)


    # 활성화 함수도 바꿔가며 실험해보자！
    # z = sigmoid(a)
    z1 = ReLU(a1)
    z2 = ReLU(a2)
    z3 = ReLU(a3)
    z4 = ReLU(a4)
    # z = tanh(a)

    activations1[i] = z1
    activations2[i] = z2
    activations3[i] = z3
    activations4[i] = z4

# 히스토그램 그리기
# for i, a1 in activations1.items():
#     plt.subplot(1, len(activations1), i+1)
#     plt.title(str(i+1) + "-layer")
#     if i != 0: plt.yticks([], [])
#     plt.xlim(0.1, 1)
#     plt.ylim(0, 7000)
#     plt.hist(a1.flatten(), 20, range=(0,1))
# plt.show()

for i, a2 in activations2.items():
    plt.subplot(1, len(activations2), i+1)
    plt.title(str(i+1) + "-layer")
    if i != 0: plt.yticks([], [])
    plt.xlim(0.1, 1)
    plt.ylim(0, 7000)
    plt.hist(a2.flatten(), 30, range=(0,1))
plt.show()

for i, a3 in activations3.items():
    plt.subplot(1, len(activations3), i+1)
    plt.title(str(i+1) + "-layer")
    if i != 0: plt.yticks([], [])
    plt.xlim(0.1, 1)
    plt.ylim(0, 7000)
    plt.hist(a3.flatten(), 15, range=(0,1))
plt.show()

for i, a4 in activations4.items():
    plt.subplot(1, len(activations4), i+1)
    plt.title(str(i+1) + "-layer")
    if i != 0: plt.yticks([], [])
    plt.xlim(0.1, 1)
    plt.ylim(0, 7000)
    plt.hist(a4.flatten(), 30, range=(0,1))
plt.show()
