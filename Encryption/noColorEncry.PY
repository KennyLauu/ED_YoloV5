import numpy as np
from Lorenz_ode45 import Lorenz_ode45
from OurEncryLifting import OurEncryLifting

def noColorEncry(img:np, key):
    w,h,z = img.shape
    s = [w,h,z] # shape
    img = img.astype(np.float64)
    P = img.T.flatten()

    Sq = ChaoSq(key, s)
    Sq = Sq.astype(np.float64)

    # 加密操作
    C = OurEncryLifting(P, Sq[:, 0], Sq[:, 1], Sq[:, 2], Sq[:, 3])
    C = C.reshape(s)
    C = C.astype(np.uint8)
    
def ChaoSq(x, s):
    '''
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
img = np.random.randn(24,24,3)
key = np.array([21, 25, 42, 168])
noColorEncry(img, key)