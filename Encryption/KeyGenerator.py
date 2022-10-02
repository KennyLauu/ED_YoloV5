import numpy
import numpy as np
import hashlib

def KeyGenerator(img):
    h = hashlib.sha256(img).hexdigest()
    n = len(h) // 2

    key = []
    for i in range(0, n):
        j = 2 * i
        key.append(int(h[j:(j + 2)], 16)) #16进制转10进制

    return np.array(key, dtype=np.int64)

# if __name__ == '__main__':
#     key = KeyGenerator('15461331')