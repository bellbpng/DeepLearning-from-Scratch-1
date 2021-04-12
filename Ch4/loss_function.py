import numpy as np

#오차제곱합 sum of squares for error
def sum_squares_error(y,t):
    return 0.5*np.sum((y-t)**2)

#교차엔트로피오차 cross_entropy_error(y,t)
#one-hot-encoding으로 이루어진 데이터인 경우 정답일 때의 추정(t=1일때, y)만을 반영하게 됨
def cross_entropy_error(y,t):
    delta = 1e-7 #0.0000001, 아주 작은 값을 더해서 log 함수에 0이 들어가는 것을 방지
    return -np.sum(t*np.log(y+delta))
