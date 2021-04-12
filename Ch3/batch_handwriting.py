import sys, os
sys.path.append(os.pardir) # 부모 디렉터리에서 파일을 불러올 수 있도록 설정
import numpy as np
import pickle # 원하는 데이터를 자료형의 변경없이 파일로 저장하여 그대로 로드할 수 있다
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax

#데이터 불러오기
def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize = True, flatten = True, one_hot_label = False)
    return x_test, t_test

#신경망 구현
def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

#예측함수
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    
    return y

x, t = get_data()
network = init_network()

batch_size = 100 # 배치크기 설정
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis = 1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size]) # 배치 단위로 분류한 결과를 실제 답과 비교

print("Accuracy: " + str(float(accuracy_cnt)/len(x)))