import sys,os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from TwoLayerNet import TwoLayerNet

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, flatten=True, one_hot_label=True)

train_loss_list = []

#하이퍼파라미터 설정
iters_num = 10000 # 반복 갱신 횟수 설정
train_size = x_train.shape[0] # 훈련데이터크기는 입력 이미지의 개수
batch_size = 100 # 미니배치 크기. 임의로 100장을 뽑는다.
learning_rate = 0.1 # 학습률. 매개변수 값을 갱신하는 정도
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    #미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size) # np.arange(train_size)에서 batch_size만큼 샘플을 뽑는다.
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    #기울기 계산
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)

    # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate*grad[key] # 경사하강법 수식 구현

    #학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
