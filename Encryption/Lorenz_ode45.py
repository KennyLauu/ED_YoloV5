import numpy as np
from scipy.integrate import solve_ivp
# import matplotlib.pyplot as plt

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
    
    step = t[1] - t[0]
    sol = solve_ivp(f_Lorenz, (min(t), max(t)), x, max_step=step, min_step=step)
    return [sol.t, sol.y.T]

    

def f_Lorenz(t, x):
    dx = [0, 0, 0, 0]
    a, b, c, r = 10, 8/3, 28, -1.312
    dx[0] = a*(x[1] - x[0]) + x[3]
    dx[1] = c*x[0] - x[1] - x[0]*x[2]
    dx[2] = x[0]*x[1] - b*x[2]
    dx[3] = -x[1]*x[2] + r*x[3]
    return dx