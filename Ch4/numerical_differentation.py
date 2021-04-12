import numpy as np
import matplotlib.pylab as plt
#수치미분
#해석적 미분과는 다르게 정확한 미분값을 구하는 데 어려움이 존재.
#아주 작은 차분으로 미분하여 오차를 최소화하는 방법
def numerical_diff(f,x):
    h = 1e-4 #0.0001
    return (f(x+h)-f(x-h))/(2*h)

#수치미분 예시
def function_1(x):
    return 0.01*x**2+0.1*x

x = np.arange(0.0, 20.0, 0.1) #0에서 20까지 0.1간격으로 나눈 배열을 형성
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x,y)
plt.show()

#기울기 - 모든 변수의 편미분을 벡터로 정리한 것
def numerical_gardient(f,x):
    h = 1e-4 #0.0001
    grad = np.zeros_like(x) #x와 형상이 같고 그 원소가 모두 0인 배열을 만듦
    
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)
        
        x[idx] = tmp_val - h
        fxh2 = f(x)
        
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val
        
    return grad
    