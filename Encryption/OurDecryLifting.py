import numpy as np

# 注：这里我改为了OurDecryLifting
def OurDecryLifting(T, S1, S2, S3, S4):
    # LIFTING SCHEME INCYRPTION
    I = T[0::2]
    Q = T[1::2]
    A = I[0::2]
    B = I[1::2]
    C = Q[0::2]
    D = Q[1::2]

    for j in range(0,4):
        H = Update(D, A, S4[len(A) - 1])
        E = Predict(A, H, S4)
        A = np.flip(E)
        D = np.flip(H)

        F = Update(B, C, S3[len(C) - 1])
        G = Predict(C, H, S3)
        C = np.flip(G)
        B = np.flip(F)

        G = Update(C, D, S2[len(D) - 1])
        H = Predict(D, G, S2)
        C, D = G, H

        E = Update(A, B, S1[len(B) - 1])
        F = Predict(B, E, S1)
        A, B = E, F

    ES = merge(A, B, C, D)
    return ES

def Predict(P, Q:np, S):
    '''
    预测
    '''
    P = np.mod((P - S[0:len(P)]), 256)
    
    # 初始化T
    T = P

    if len(P) > len(Q):
        # ⚠ 注：这里可能有问题 超出索引
        Q[len(P) - 1] = S[len(P) - 1]
    elif len(P) == len(Q):
        # ⚠ 注：这里可能有问题
        Q = np.append(Q, Q[0])
    else:
        Q[len(P)] = Q[0]

    for i in range(0,len(P)):
        T[i] = np.bitwise_xor(int(P[i]), int(np.floor(0.5*(Q[i] + Q[i+1]))))

    return T

def Update(P, Q, S):
    '''
    更新
    '''
    if len(P) > len(Q):
        # ⚠ 注：可能有问题 超出索引
        Q[len(P) - 1] = S
    
    W, InitW, InitQ = [], 0, 0
    W.append(np.mod(P[0] - np.floor(0.25*(InitQ + Q[0] + 2)) - InitW, 256))
    for i in range(1, len(P)):
        W.append(np.mod(P[i] - np.floor(0.25*(Q[i-1] + Q[i] + 2)) - P[i-1], 256))

    return np.array(W)

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