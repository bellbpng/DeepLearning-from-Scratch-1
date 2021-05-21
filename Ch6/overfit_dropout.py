import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.trainer import Trainer # 신경망 훈련을 대신해주는 클래스

'''
드롭아웃(Dropout) 구현. (common.layers에 class로 구현되어있음)
class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    ---훈련시에는 순전파때마다 self.mask에 삭제할 뉴런을 False로 표시한다.---
    def forward(self, x, train_flg=True):
        if train_flg:
            --- self.mask에 x와 형상이 같은 배열을 무작위로 생성하고 dropout_ration보다 큰 원소만 True로 설정한다. ---
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x*self.mask
        else:
            return x*(1.0-self.dropout_ratio)

    --- 역전파는 ReLU와 같은 동작을 한다. 순전파때 신호를 통과시킨 뉴런은 역전파때도 그대로 통과시키고,
        그렇지 않은 뉴런은 역전파때 신호를 차단한다. ---
    def backward(self, dout):
        return dout*self.mask
'''

# 데이터 불러오기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 오버피팅을 재현하기 위해 훈련데이터를 줄임
x_train = x_train[:300]
t_train = t_train[:300]

# 드롭아웃 사용 유무와 비울 설정 ========================
use_dropout = True  # 드롭아웃을 쓰지 않을 때는 False
dropout_ratio = 0.2

# 신경망 학습
network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10,
                            use_dropout=True, dropout_ration = dropout_ratio)
trainer = Trainer(network, x_train, t_train, x_test, t_test, epochs=301, mini_batch_size=100,
                    optimizer='sgd', optimizer_param={'lr':0.01}, verbose=True)
trainer.train()

train_acc_list, test_acc_list = trainer.train_acc_list, trainer.test_acc_list

# 그래프 그리기
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
