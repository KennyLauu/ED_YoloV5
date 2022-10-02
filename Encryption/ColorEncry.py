'''
NOT USED - RESERVED    
AUTHOR: LONGBEFER
'''

import numpy as np

# 导入本地文件
# from NewColorEncry import 

def ColorEncry(img:np):
    '''
    对彩色图像使用提升方案加密
    '''
    w,h,z = img.shape
    s = [w, h, z] # shape
    
    # 转换img为float64类型
    print(img.dtype)
    img = img.astype(np.float64)
    
    Ball = Block(img[:,:,0])
    Ball[4:7] = Block(img[:,:,1])
    Ball[8:11] = Block(img[:,:,2])
    
    # 混沌序列
    x0, u = 0.73, 3.674
    S = logistic(x0, u, 50+200)
    S = S.T
    # 注意索引 S[]
    A = np.mod(np.floor(S[200:50+200+1]*(10**6)), 12) + 1
    Sbox = np.empty((0,0))
    i, n = 0, 0
    
    x = [21, 25, 42, 168]
    Sq = ChaoSq(x, s)
    Sq = Sq.astype(np.float64)

    # note 此处的循环可能不正确，由于matlab是
    # 从1开始的，而python是从0开始的
    while n <= 12:
        i += 1
        if ismember(A[i], Sbox): # 暂时无法找到ismember的替代方案
            continue
        else: 
            Sbox[n] = A[i]
            n += 1

    # 加密操作
    for v in range(0,3):
        pass


def Block(Bimg):
    '''
    RGB 分量分块处理
    '''
    # 每隔2行取一个值，再转置
    F = Bimg[0::2,:].T.flatten()
    G = Bimg[1::2,:].T.flatten()

    B = []
    B.append(F[0::2])
    B.append(F[1::2])
    B.append(G[0::2])
    B.append(G[1::2])

    return B

def merge(W, D, Y, C):
    F = combine(W, D)
    G = combine(Y, C)
    Q = combine(F, G)
    return Q

def combine(A, B):
    pass

def ChaoSq(x, s):
    '''
    对超混沌系统生成的序列进行处理
    '''
    N = s[0] * s[1]
    SqN = 201 + np.ceil(N/4) * 4 # 注意索引
    L,T,t = 0.01, (SqN)*L, range(L, T+L, L) 
    [_, xn] = Lorenz_ode45(t, x)

    # 注意索引
    A1 = np.mod(np.round(xn[200:SqN, 1]*(10**6)), 256)
    A2 = np.mod(np.round(xn[200:SqN, 2]*(10**6)), 256)
    A3 = np.mod(np.round(xn[200:SqN, 3]*(10**6)), 256)
    A4 = np.mod(np.round(xn[200:SqN, 4]*(10**6)), 256)
    Sq = np.array([A1, A2, A3, A4])
    Sq = Sq.T.reshape((-1,16)) # 不一定正确！

    return Sq


# for test
img = np.random.randn(24,24,3)
a = ColorEncry(img)