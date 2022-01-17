#MNIST 데이터셋을 읽어오는 코드
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)
    
print(x_train.shape) #60000행, 784열의 훈련데이터
print(t_train.shape) #60000행, 10열의 훈련 정답레이블


train_size = x_train.shape[0] #60000개의 데이터
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size) #(60000,10) 0이상 60000미만 수 중 10개를 무작위로 추출
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

# #배치용 교차엔트로피 구현
def cross_entropy_error(y, t):
    delta = 1e-7
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1) 
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size

"""
t.size == y.size 가 원-핫 인코딩임을 확인할 수 있는 이유
- t는 정답레이블을 배열로 가지는데 원-핫 인코딩이 되어있지 않은 경우 말 그대로 '정답'만 원소로 가진다.
따라서, t = [2 3 5 6 1 9 7 3 2 1] 와 같이 1차원배열의 양상을 띄게 되므로 batch_size가 10인 경우
t.size = 10 이다.
- 만약 원-핫 인코딩이 되어있다면 t는 정답레이블의 인덱스만 1이고 나머지는 0인 배열을 원소로 가진다.
따라서, t = [[0 1 0 0 0 0 0 0 0 0], [0 0 1 0 0 0 0 0 0 0], ... ,[0 1 1 0 0 0 0 0 0 0]] 와 같은 형태로
t.size = 10x10 = 100 = y.size 이다.
- y[np.arange(batch_siz), t] 는 t가 원-핫 인코딩이 되어있다면 t = t.argmax(axis=1)에서 1을 원소로하는 
인덱스만을 가지는 1차원 배열이 만들어지므로 결과는 y[0, 2], y[1, 3], y[2, 5] ... 를 원소로 가지는 배열이 형성된다.
"""

