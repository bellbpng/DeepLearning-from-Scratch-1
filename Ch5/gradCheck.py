import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from twoLayerNet import TwoLayerNet

#데이터 읽어오기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size = 784, hidden_size = 50, output_size = 10)

x_batch = x_test[:3] # x_test[0]~x_test[2]까지 요소를 가지는 리스트
t_batch = t_test[:3] # t_test[0]~t_text[2]까지 요소를 가지는 리스트

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backp = network.gradient(x_batch, t_batch)

# 각 가중치의 차이의 절댓값을 구한 후, 그 절댓값들의 평균을 구한다.
for key in grad_numerical.keys():
    diff = np.average( np.abs(grad_backp[key] - grad_numerical[key]) )
    print(key + ":" + str(diff))