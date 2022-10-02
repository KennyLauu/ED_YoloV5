import numpy
import numpy as np
import hashlib


def KeyGenerator(img):
    h = hashlib.sha256(img.encode('utf-8')).hexdigest()
    n = len(h) // 2
    h = np.array(list(h))
    x = h.shape
    key = []

    key = np.zeros(shape=n, dtype=numpy.int16)
    for i in range(0, n):
        j = 2 * i
        """
            由于预处理时用的是int类型，所以选择将list的值拆出来
            提取出来的每一个值是16进制的，转为十进制，则是通过 a * 16 + b
        """
        a = h[j]
        b = h[j + 1]

        # 检验值是否错误
        # a1 = int(h[j], 16)
        # b1 = int(h[j + 1], 16)

        # 将值转换为10进制
        c = int(h[j], 16) * 16 + int(h[j + 1], 16)
        key[i] = c

    # 从行向量转换为列向量
    key = key.reshape((n, 1))

    return key


if __name__ == '__main__':
    key = KeyGenerator('15461331')
    print(key)
