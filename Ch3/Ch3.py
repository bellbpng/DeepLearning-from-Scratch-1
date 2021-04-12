import numpy as np
import matplotlib.pylab as plt

# 계단함수 구현
def step_function(x):
    return np.array(x > 0, dtype=np.int)


x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)

plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.ylim(-0.1, 1.1)
plt.show()

# 시그모이드함수 구현
def sigmoid(x):
    return 1/(1+np.exp(-x))


x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

# ReLU함수 구현
def relu(x):
    return np.maximum(0, x)


x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)
plt.plot(x, y)
plt.show()


X = np.array([1.0, 0.5])  # 1*2
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])  # 2*3
B1 = np.array([0.1, 0.2, 0.3])  # 1*3

print(W1.shape)
print(X.shape)
print(B1.shape)

A1 = np.dot(X, W1) + B1  # 1*3
Z1 = sigmoid(A1)  # 1*3

print(A1)
print(Z1)

W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.5]])  # 3*2
B2 = np.array([0.1, 0.2])  # 1*2

print(Z1.shape)
print(W2.shape)
print(B2.shape)

A2 = np.dot(Z1, W2) + B2  # 1*2
Z2 = sigmoid(A2)  # 1*2

# 항등함수. 출력층의 활성화 홤수를 표현
def identity_function(x):
    return x


W3 = np.array([[0.1, 0.3], [0.2, 0.4]])  # 2*2
B3 = np.array([0.1, 0.2])  # 1*2
A3 = np.dot(Z2, W3) + B3  # 1*2
Y = identity_function(A3)

print(Y)

# 3층 신경망 구현 정리
def init_network():
    network = {}  # 딕셔너리 선언
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

# 신호의 순전파(입력에서 출력방향)를 구현
def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y


network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)


# 소프트맥스 함수 구현 - 오버플로 발생
# def softmax(a):
#     exp_a = np.exp(a) # 지수 함수
#     sum_exp_a = np.sum(exp_a) # 지수 함수 합
#     return y

# 소프트맥스 함수 재구현
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)  # 오버플로 방지
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y
