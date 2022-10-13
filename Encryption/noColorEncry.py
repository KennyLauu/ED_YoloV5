import numpy as np
import sys
from pathlib import Path
import os

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from EncryUtils import Lorenz_ode45
from OurEncryLifting import OurEncryLifting


def noColorEncry(img: np, key):
    '''
    对彩色图像使用提升方案加密
    '''
    w, h, z = img.shape
    s = [w, h, z]  # shape
    img = img.astype(np.float64)
    P = img.T.flatten()

    Sq = ChaoSq(key, s)
    Sq = Sq.astype(np.float64)

    # 加密操作
    C = OurEncryLifting(P, Sq[:, 0], Sq[:, 1], Sq[:, 2], Sq[:, 3])
    C = C.reshape((z, w, h)).T
    C = C.astype(np.uint8)
    return C


def ChaoSq(x, s):
    '''
    对超混沌系统生成的序列进行处理
    '''
    assert (len(s) == 3), 'error s must be 3 dimension but get %d dimension' % len(s)

    N = s[0] * s[1] * s[2]
    SqN = int(200 + np.ceil(N / 4))
    L = 0.01
    T = SqN * L
    t = np.arange(L, T + L, L)
    [_, xn] = Lorenz_ode45(t, x.astype(np.float64))

    A1 = np.mod(np.round(xn[200:SqN, 0] * (10 ** 6)), 256)
    A2 = np.mod(np.round(xn[200:SqN, 1] * (10 ** 6)), 256)
    A3 = np.mod(np.round(xn[200:SqN, 2] * (10 ** 6)), 256)
    A4 = np.mod(np.round(xn[200:SqN, 3] * (10 ** 6)), 256)
    Sq = np.array([A1, A2, A3, A4])

    return Sq.T
