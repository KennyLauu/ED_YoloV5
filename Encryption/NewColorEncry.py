'''
NOT USED - RESERVED 
AUTHOR: LONGBEFER
'''
import numpy as np
from EncryUtils import Lorenz_ode45, logistic
from OurEncryLifting import OurEncryLifting

def NewColorEncry(img:np, key):
    '''
    对彩色图像使用提升方案加密
    '''
    w,h,z = img.shape
    s = [w,h,z] # shape
    img = img.astype(np.float64)
    Ball = Block(img)

    # 混沌序列
    x0, u = .73, 3.674
    S = logistic(x0, u, 50 + 200)
    S = S.T
    Sq = ChaoSq(key, s)
    Sq = Sq.astype(np.float64)



    # 加密操作
    # C = OurEncryLifting(P, Sq[:, 0], Sq[:, 1], Sq[:, 2], Sq[:, 3])
    C = C.reshape((z,w,h)).T
    C = C.astype(np.uint8)
    return C
    
def ChaoSq(x, s):
    '''
    对超混沌系统生成的序列进行处理
    '''

    N = s[0]*s[1]
    SqN = int(200 + np.ceil(N/4)*4)
    L = 0.01
    T = SqN*L
    t = np.arange(L, T+L, L)
    [_, xn] = Lorenz_ode45(t, x)

    # 未知原因，导致xn会多一行，暂时不修改
    A1 = np.mod(np.round(xn[200:SqN, 0]*(10**6)), 256)
    A2 = np.mod(np.round(xn[200:SqN, 1]*(10**6)), 256)
    A3 = np.mod(np.round(xn[200:SqN, 2]*(10**6)), 256)
    A4 = np.mod(np.round(xn[200:SqN, 3]*(10**6)), 256)
    Sq = np.array([A1, A2, A3, A4]).T
    # ⚠ 注：未验证，可能不正确
    Sq = np.reshape(Sq, (-1, 16))

    return Sq

def Block(Bimg):
    '''
    按索引奇偶性分块
    '''

    E = Bimg[0::2,:].T.flatten()
    F = Bimg[1::2,:].T.flatten()
    return [E, F]

def AllBlock(img):
    [P, T] = Block(img)
    [X1, Y1] = Block(P)
    [X2, Y2] = Block(T)

    # ⚠ 注：未标明B
    assert 0, 'error undefine B'
    # return B

def merge(W, D, Y, C):
    '''
    合并
    '''
    F = combine(W, D)
    G = combine(Y, C)
    Q = combine(F, G)
    return Q

def combine(A, B):
    if len(A) > len(B):
        # ⚠ 注：未验证，可能有错误
        R = np.array([A[0:len(B)], B])
        R = R.T
        R = R.flatten()
        R = np.append(R, A[-1])
    elif len(A) < len(B):
        # ⚠ 注：未验证，可能有错误
        R = np.array([A, B[0:len(A)]])
        R = R.T
        R = R.flatten()
        R = np.append(R, B[-1])
    else:
        R = np.array([A, B])
        R = R.T
        R = R.flatten()
    
    return R



# for test
# img = np.array([[[1,2,3],[4,5,6],[7,8,9],[10,11,12]],
#                 [[13,14,15],[16,17,18],[19,20,21],[22,23,24]],
#                 [[25,26,27],[28,29,30],[31,32,33],[34,35,36]],
#                 [[37,38,39],[40,41,42],[43,44,45],[46,47,48]]])
# # img = np.random.randint(0, 256, size=[24,24,3])
# key = np.array([21, 25, 42, 168])
# noColorEncry(img, key)