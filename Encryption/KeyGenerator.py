import numpy
import numpy as np
import hashlib

def KeyGenerator(img):
    h = hashlib.sha256(img.encode('utf-8')).hexdigest()
    n = len(h) // 2
    h = np.array(h)

    key = np.zeros(shape=(n, 1), dtype=numpy.int8)
    for i in range(1, n):
        j = (i - 1) * 2 + 1
        key[i] = int(h[j, j + 1], 16) #16进制转10进制

    return key

if __name__ == '__main__':
    key = KeyGenerator('15461331')