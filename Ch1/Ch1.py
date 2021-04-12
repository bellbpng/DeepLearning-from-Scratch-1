from matplotlib.image import imread
import numpy as np
import matplotlib.pyplot as plt

# 데이터 준비
x = np.arange(0, 6, 0.1)  # 0부터 6까지 0.1 간격으로 원소를 형성
y1 = np.sin(x)  # x의 원소를  sin 함수에 대입
y2 = np.cos(x)

# 그래프 그리기
plt.plot(x, y1, label="sin")  # plot 메서드를 호출해 그래프를 그림
plt.plot(x, y2, label="cos")
plt.xlabel("x")  # x축 이름
plt.ylabel("y")  # y축 이름
plt.title('sin & cos')  # 제목
plt.legend()  # 범례
plt.show()  # 그린 그래프를 보여줌


img = imread('background.jpeg')  # 이미지 읽어오기, 해당 디렉토리에 없다면 상세경로 지정

plt.imshow(img)
plt.show()
