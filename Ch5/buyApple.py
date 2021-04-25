class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None
    
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x*y

        return out

    def backward(self, dout):
        dx = dout*self.y
        dy = dout*self.x

        return dx, dy

apple = 100
apple_num = 2
tax = 1.1

#계층 구현
apple_layer = MulLayer()
tax_layer = MulLayer()

#forward propagation
apple_price = apple_layer.forward(apple, apple_num)
price = tax_layer.forward(apple_price, tax)

print("총 지출: ", price)

#backward propagation
dprice = 1
dapple_price, dtax = tax_layer.backward(dprice)
dapple, dapple_num = apple_layer.backward(dapple_price)

print("사과 가격에 대한 미분: ", dapple)
print("사과 개수에 대한 미분: ", dapple_num)
print("소비세에 대한 미분: ", dtax)
