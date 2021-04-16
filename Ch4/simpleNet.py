import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

# 신경망 구성 for 가중치 설정, 예측, 손실함수 값 계산
class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t): # x는 입력데이터, t는 정답레이블
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

net = simpleNet()
print("가중치 값")
print(net.W)
print()

x = np.array([0.6, 0.9])
p = net.predict(x)
print("예측값")
print(p)
print()

print("최대값 인덱스") # 최대값의 인덱스
print(np.argmax(p))
print()

t = np.array([0, 0, 1])
print("손실함수 값")
print(net.loss(x,t))
print()

'''
def f(w):
    return net.loss(x,t)
'''

#람다(lambda) 표현식으로 재구현
f = lambda w: net.loss(x,t) 
dW = numerical_gradient(f, net.W) # 신경망(net)의 가중치를 함수 f의 인수로 받아 수치미분을 진행한다. 

print("수치미분 결과")
print(dW)

