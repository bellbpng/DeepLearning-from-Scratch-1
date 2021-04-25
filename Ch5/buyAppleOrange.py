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

class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x+y
        return out

    def backward(self, dout):
        dx = dout*1
        dy = dout*1

        return dx,dy

apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

#계층 설정
apple_layer = MulLayer()
orange_layer = MulLayer()
add_layer = AddLayer()
tax_layer = MulLayer()

#forward propagation
apple_price = apple_layer.forward(apple_num, apple)
orange_price = orange_layer.forward(orange, orange_num)
fruit_price = add_layer.forward(apple_price, orange_price)
price = tax_layer.forward(fruit_price, tax)

print("사과 가격: ", apple_price)
print("귤 가격: ", orange_price)
print("순수 과일 가격: ", fruit_price)
print("총 지출: {:.0f}".format(price))

#backward propagation
dprice = 1
dfruit_price, dtax = tax_layer.backward(dprice)
dapple_price, dorange_price = add_layer.backward(dfruit_price)
dapple_num, dapple = apple_layer.backward(dapple_price)
dorange, dorange_num = orange_layer.backward(dorange_price)

print("사과의 개수에 대한 미분: {:.0f}".format(dapple_num))
print("사과 가격에 대한 미분: ", dapple)
print("귤의 개수에 대한 미분: {:.0f}".format(dorange_num))
print("귤 가격에 대한 미분: {:.0f}".format(dorange))
print("소비세에 대한 미분: ", dtax)
