import numpy as np

# 注：这里我改为了OurDecryLifting
def OurDecryLifting(T, S1, S2, S3, S4):
    # LIFTING SCHEME INCYRPTION
    I = T[1::2]
    Q = T[0::2]
    A = I[1::2]
    B = I[0::2]
    C = Q[1::2]
    D = Q[0::2]

    for j in range(0,4):
        H = Update(D, A, S4[len(D) - 1])
        E = Predict(A, H, S4)
        A = np.flip(E)
        D = np.flip(H)

        F = Update(B, C, S3[len(B) - 1])
        G = Predict(C, F, S3)
        C = np.flip(G)
        B = np.flip(F)

        G = Update(C, D, S2[len(C) - 1])
        H = Predict(D, G, S2)
        C, D = G, H

        E = Update(A, B, S1[len(A) - 1])
        F = Predict(B, E, S1)
        A, B = E, F

    ES = merge(A, B, C, D)
    return ES

def Predict(P, Q:np, S):
    '''
    预测
    '''
    P = np.mod((P - S[0:len(P)]), 256)
    T = P

    # match dimension
    if len(P) > len(Q):
        Q = np.append(Q, S[len(P) - 1])
    
    # shift
    A = np.zeros(P.shape)
    A[:-1] = Q[1:len(P)] # 将Q向上移动
    A[-1] = Q[0] 

    T = np.bitwise_xor(P.astype(np.uint8), np.floor(.5*(Q[:len(P)] + A)).astype(np.uint8))
    
    return T

def Update(P, Q, S):
    '''
    更新
    '''
    # match dimension
    if len(P) > len(Q):
        Q = np.append(Q, S)
    
    W, InitP, InitQ = np.zeros(P.shape), 0, 0

    W[0] = np.mod(P[0] - np.floor(0.25*(InitQ + Q[0] + 2)) - InitP, 256)
    W[1:] = np.mod(P[1:] - np.floor(.25*(Q[:(len(P) - 1)] + Q[1:len(P)] + 2)) - P[:-1], 256)

    return W

def merge(W, D, Y, C):
    '''
    合并
    '''
    F = combine(D, W)
    G = combine(C, Y)
    Q = combine(G, F)
    return Q

def combine(A, B):
    if len(A) > len(B):
        R = np.array([A[0:len(B)], B])
        R = R.T
        R = R.flatten()
        R = np.append(R, A[-1])
    elif len(A) < len(B):
        R = np.array([A, B[0:len(A)]])
        R = R.T
        R = R.flatten()
        R = np.append(R, B[-1])
    else:
        R = np.array([A, B])
        R = R.T
        R = R.flatten()
    
    return R