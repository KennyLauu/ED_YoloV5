import numpy as np
import hashlib
from scipy.integrate import solve_ivp
import numba

def KeyGenerator(img):
    h = hashlib.sha256(img).hexdigest()
    n = len(h) // 2

    key = []
    for i in range(0, n):
        j = 2 * i
        key.append(int(h[j:(j + 2)], 16)) #16进制转10进制

    return np.array(key, dtype=np.int64)


def ProcessingKey(img):
    '''
    生成密钥并进行处理
    '''
    K = []
    key = KeyGenerator(img)
    
    K.append(np.mod((np.bitwise_xor(key[ 3], key[ 4]) + np.bitwise_xor(key[ 5], key[ 6]) + np.bitwise_xor(key[ 7], key[ 8]) + np.bitwise_xor(key[ 9], key[10])), 41))
    K.append(np.mod((np.bitwise_xor(key[10], key[11]) + np.bitwise_xor(key[12], key[13]) + np.bitwise_xor(key[14], key[15]) + np.bitwise_xor(key[16], key[17])), 41))
    K.append(np.mod((np.bitwise_xor(key[17], key[18]) + np.bitwise_xor(key[19], key[20]) + np.bitwise_xor(key[21], key[22]) + np.bitwise_xor(key[23], key[24])), 80) + 1)
    K.append(np.mod((np.bitwise_xor(key[24], key[25]) + np.bitwise_xor(key[26], key[27]) + np.bitwise_xor(key[28], key[29]) + np.bitwise_xor(key[30], key[31])), 251))

    return np.array(K)


def logistic(x0, u, n):
    '''
    0 < x0 < 1 和
    3.5699456 < u < 4 时
    处于混沌状态
    '''
    x = [0 for _ in range(n)]
    x[0] = x0
    for i in range(1, n):
        x[i] = u * x[i - 1] * (1 - x[i - 1])
    
    return np.array(x)


def Lorenz_ode45(t, x):
    '''
    LORENZ_ODE45 使用函数ode45直接求解微分方程
    xn是一个n行4列的方程,n为t中的元素
    Lorenz超混沌系统
    xn是一个n行4列的方程,n为t中的元素
    Lorenz超混沌系统初始值取值范围为：x0:(-40,40) y0:(-40,40) z0:(1,81) w0:(-250,250) 
    a=10 b=8/3 c=28 r:(-1.52,-0.06)
    '''

    # [tn, xn] = ode45(f_Lorenz, t, x)
    
    # step = t[1] - t[0]
    # sol = solve_ivp(f_Lorenz, (min(t), max(t)), x, max_step=step, min_step=step)
    import time
    start = time.time()
    sol = solve_ivp(f_Lorenz, (np.min(t), np.max(t)), y0=x, t_eval=t)
    print('solve ivp spend ', time.time() - start)
    return [sol.t, sol.y.T]

# 自定义f_Lorenz函数
# @numba.jit(nopython=True)
def f_Lorenz(t, x):
    dx = [0, 0, 0, 0]
    a, b, c, r = 10, 8/3, 28, -1.312
    dx[0] = a*(x[1] - x[0]) + x[3]
    dx[1] = c*x[0] - x[1] - x[0]*x[2]
    dx[2] = x[0]*x[1] - b*x[2]
    dx[3] = -x[1]*x[2] + r*x[3]
    return dx