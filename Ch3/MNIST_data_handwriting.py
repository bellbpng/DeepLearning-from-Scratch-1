import pickle
from PIL import Image  # Python Image Library
import numpy as np
import sys
import os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져오도록 설정
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax

def img_show(img):
    # 넘파이로 저장된 이미지 데이터를 PIL용 데이터 객체로 변환
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


# (훈련이미지, 훈련레이블), (시험이미지, 시험레이블) 형식으로 반환
(x_train, t_train), (x_test, t_test) = \
    load_mnist(
        flatten=True, normalize=False)  # flatten = True, 읽어들인 이미지는 1차원 넘파이 배열로 저장됨

# print(x_train.shape)
# print(t_train.shape)
# print(x_test.shape)
# print(t_test.shape)

img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28, 28)  # img를 표시하기 위해 원래 형상으로 다시 변형
print(img.shape)

img_show(img)

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정

# 신경망의 추론 처리


def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, flatten=True,
                   one_hot_label=False)  # 입력 데이터를 정규화. 데이터 전처리 작업
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y  # 각 레이블의 확률을 넘파이 배열로 반환

x, t = get_data()


network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)  # 확률이 가장 높은 원소의 인덱스 반환
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt / len(x))))
