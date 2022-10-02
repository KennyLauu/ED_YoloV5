

def OurEncryLifting(T, S1, S2, S3, S4):
    # LIFTING SCHEME INCYRPTION
    I = T[1::2]
    Q = T[0::2]
    A = I[1::2]
    B = I[0::2]
    C = Q[1::2]
    D = Q[0::2]

    for j in range(0,4):
        F = Predict(B, A, S1)


    pass

def Predict(P, Q, S):
    '''
    预测
    '''
    pass

def Update(P, Q, S):
    '''
    更新
    '''
    pass

def merge(W, D, Y, C):
    '''
    合并
    '''
    pass
