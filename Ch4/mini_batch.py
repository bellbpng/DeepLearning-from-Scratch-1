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

#배치용 교차엔트로피 구현
def cross_entropy_error(y,t):
    if y.ndim==1 : # y가 1줄짜리 데이터라면(행의 개수가 1)
        t = t.reshape(1, t.size) #size는 전체 원소의 개수를 반환한다.
        y = y.reshape(1, y.size)
    batch_size_entropy = y.shape[0] #y의 행의개수(데이터의 개수)
    return -np.sum(t*np.log(y+1e-7))/batch_size_entropy

#정답레이블이 원-핫 인코딩이 아닌 경우 배치용 교차엔트로피
def cross_entropy_error2(y,t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    batch_size_entropy = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size_entropy),t] + 1e-7)) / batch_size_entropy


"""
np.log(y[np.arange(batch_size_entropy), t])에 대한 설명
- np.arange(batch_size)는 0부터 (batch_size_entropy-1) 까지 1간격으로 배열을 생성한다.
- if batch_size_entropy == 5 then [0, 1, 2, 3, 4]
- t 에는 [2,7,0,9,4]와 같이 레이블이 담겨있으므로, 
- y[np.arange(batch_size_entropy), t]는 각 데이터의 정답 레이블에 해당하는 신경망의 출력을 추출한다.
- 위의 예에서는 y[0,2], y[1,7], y[2,0], y[3,9], y[4,4]인 넘파이 배열이 형성된다.
"""
