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

def tangent_line(f, x):
    d = numerical_diff(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y
'''
사용된 람다 표현식 정리 (lambda)
- plus_ten = lambda x: x+10
=> 매개변수 x를 전달받아 10을 더한 값을 반환하는 익명함수로 활용
=> plus_ten(20), plus_ten(11) 와 같이 람다 표현식이 할당된 변수에 매개변수를 전달해주는 형태로 호출

'''
x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")

tf = tangent_line(function_1, 5)
y2 = tf(x) # 람다식에 매개변수를 전달

plt.plot(x, y)
plt.plot(x, y2)
plt.show()


#기울기 - 모든 변수의 편미분을 벡터로 정리한 것
def _numerical_gradient_no_batch(f,x):
    h = 1e-4 #0.0001
    grad = np.zeros_like(x) #x와 형상이 같고 그 원소가 모두 0인 배열을 만듦
    
    # x의 각 원소에 대해서 수치미분을 구한다.
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)
        
        x[idx] = tmp_val - h
        fxh2 = f(x)
        
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val
        
    return grad

def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)
        
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)
        
        return grad

def function_2(x):
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)


def tangent_line(f, x):
    d = numerical_gradient(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y
    
if __name__ == '__main__':
    x0 = np.arange(-2, 2.5, 0.25)
    x1 = np.arange(-2, 2.5, 0.25)
    X, Y = np.meshgrid(x0, x1)
    
    X = X.flatten()
    Y = Y.flatten()
    
    grad = numerical_gradient(function_2, np.array([X, Y]) )
    
    plt.figure()
    plt.quiver(X, Y, -grad[0], -grad[1],  angles="xy",color="#666666")#,headwidth=10,scale=40,color="#444444")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.legend()
    plt.draw()
    plt.show()

