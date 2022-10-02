import numpy as np
from Lorenz_ode45 import Lorenz_ode45
from OurDecryLifting import OurDecryLifting

# 注：这里改为了noColorDecry
def noColorDecry(img:np, key):
    '''
    对彩色图像使用提升方案加密
    '''
    w,h,z = img.shape
    s = [w,h,z] # shape
    img = img.astype(np.float64)
    P = img.T.flatten()

    # 混沌序列
    Sq = ChaoSq(key, s)
    Sq = Sq.astype(np.float64)

    # for test
    # Sq = np.array([[235,20,56,144,50,13,239,127,16,166,22,147],
    # [172,71, 84, 173, 157, 228, 180, 178, 47, 13, 46, 33],
    # [196, 243,50,141,167,190,168,215,73,178,6,150],
    # [193,20,193,218,22,201,237,27,13,237,5,236]])
    # Sq = Sq.T

    # 解密操作
    C = OurDecryLifting(P, Sq[:, 0], Sq[:, 1], Sq[:, 2], Sq[:, 3])
    C = C.reshape((z,w,h)).T
    C = C.astype(np.uint8)
    return C
    
def ChaoSq(x, s):
    '''
    对超混沌系统生成的序列进行处理
    '''
    assert (len(s) == 3), 'error s must be 3 dimension but get %d dimension' %len(s)

    N = s[0]*s[1]*s[2]
    SqN = int(200 + np.ceil(N/4))
    L = 0.01
    T = SqN*L
    t = np.arange(L, T+L, L)
    [_, xn] = Lorenz_ode45(t, x)

    # 未知原因，导致xn会多一行，暂时不修改
    A1 = np.mod(np.round(xn[200:SqN, 0]*(10**6)), 256)
    A2 = np.mod(np.round(xn[200:SqN, 1]*(10**6)), 256)
    A3 = np.mod(np.round(xn[200:SqN, 2]*(10**6)), 256)
    A4 = np.mod(np.round(xn[200:SqN, 3]*(10**6)), 256)
    Sq = np.array([A1, A2, A3, A4])

    return Sq.T

# for test
# img = np.array([[[1,2,3],[4,5,6],[7,8,9],[10,11,12]],
#                 [[13,14,15],[16,17,18],[19,20,21],[22,23,24]],
#                 [[25,26,27],[28,29,30],[31,32,33],[34,35,36]],
#                 [[37,38,39],[40,41,42],[43,44,45],[46,47,48]]])
# # img = np.random.randint(0, 256, size=[24,24,3])
# key = np.array([21, 25, 42, 168])
# noColorDecry(img, key)