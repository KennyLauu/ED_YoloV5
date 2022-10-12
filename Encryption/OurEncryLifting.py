import numpy as np
import numba


def OurEncryLifting(T, S1, S2, S3, S4):
    # LIFTING SCHEME INCYRPTION
    I = T[1::2]
    Q = T[0::2]
    A = I[1::2]
    B = I[0::2]
    C = Q[1::2]
    D = Q[0::2]

    for j in range(0, 4):
        F = Predict(B, A, S1)
        E = Update(A, F, S1[len(A) - 1])
        A = np.flip(E)
        B = np.flip(F)

        H = Predict(D, C, S2)
        G = Update(C, H, S2[len(C) - 1])
        C = np.flip(G)
        D = np.flip(H)

        E = Predict(A, D, S4)
        H = Update(D, E, S4[len(D) - 1])
        A, D = E, H

        G = Predict(C, B, S3)
        F = Update(B, G, S3[len(B) - 1])
        C, B = G, F

    ES = merge(D, C, B, A)
    return ES


def Predict(P, Q: np, S):
    '''
    预测
    '''
    # match dimension
    if len(P) > len(Q):
        Q = np.append(Q, S[len(P) - 1])

    # shift
    A = np.zeros(P.shape)
    A[:-1] = Q[1:len(P)]  # 将Q向上移动
    A[-1] = Q[0]

    P = np.bitwise_xor(P.astype(np.uint8), np.floor(.5 * (Q[:len(P)] + A)).astype(np.uint8))
    T = np.mod((P + S[0:len(P)]), 256)

    return T


@numba.jit(nopython=True)
def Update(P, Q, S):
    '''
    更新
    '''
    # match dimension
    if len(P) > len(Q):
        Q = np.append(Q, S)

    W, InitW, InitQ = np.zeros(P.shape), 0, 0

    W[0] = np.mod(P[0] + np.floor(0.25 * (InitQ + Q[0] + 2)) + InitW, 256)
    for i in range(1, len(P)):
        W[i] = np.mod(P[i] + np.floor(0.25 * (Q[i - 1] + Q[i] + 2)) + W[i - 1], 256)

    return W


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
