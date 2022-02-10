# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 20:18:34 2022

@author: bokyoung
"""

#%% im2col 함수 사용해보기 (4차원 데이터 -> 2차원 배)
import sys, os
sys.path.append(os.pardir)
from common.util import im2col
import numpy as np

x1 = np.random.rand(1,3,7,7) # (데이터 수, 채널 수, 행, 열)
col1 = im2col(x1, 5, 5, stride=1, pad=0)
print(col1.shape) # (9,75), 2차원 원소는 75개로 (채널수(3) * 필터의 원소 수(5*5))

x2 = np.random.rand(10,3,7,7)
col2 = im2col(x2, 5, 5, stride=1, pad=0)
print(col2.shape) #(90,75)

#%% 합성곱 계층 구현
import sys,os
sys.path.append(os.pardir)
import numpy as np
from common.util import im2col

class Convolution:
    def __init__(self, W, b, stride = 1, pad = 0): # W: 필터(가중치), b: 편향
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        
    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = self.x.shape
        out_h = int(1+(H+2*self.pad-FH)/self.stride)
        out_w = int(1+(W+2*self.pad-FW)/self.stride)
        
        
        col = im2col(x, FH, FW, self.stride, self.pad) # 필터링하기 좋게 입력데이터를 전개
        
        # reshape 두번째 인수를 -1로 지정하면 변환 후 원소수가 유지되도록 적절하게 묶음
        # 필터를 1열로 전개하고
        col_W = self.W.reshape(FN, -1).T 
        out = np.dot(col, col_W) + self.b
        
        out = out.reshape(N, out_h, out_w, -1).transpose(0,3,1,2) # 출력 데이터를 적절한 형상으로 바꿔줌
        
        return out
    
    

#%% 풀링 계층 구현
#(1) 입력데이터 전개 -> (2) 행별 최댓값 찾기 -> (3) 적절한 모양으로 성형
import sys,os
sys.path.append(os.pardir)
import numpy as np
from common.util import im2col

class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
    def forward(self, x):
        N,C,H,W = x.shape
        out_h = int(1+(H-self.pool_h)/self.stride)
        out_w = int(1+(W-self.pool_w)/self.stride)
        
        # 전개(1)
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)
        
        # 최댓값(2)
        out = np.max(col, axis=1)
        
        # 성형(3)
        out = out.reshape(N, out_h, out_w, C).transpose(0,3,1,2)
        
        return out
        

