# 加密/解密测试

import numpy as np
# import cv2 
import matplotlib.pyplot as plt

from KeyGenerator import KeyGenerator
from noColorDecry import noColorDecry
from noColorEncry import noColorEncry


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


# img = cv2.imread('D:/User/Documents/Code/Encryption/Transfrom/ED_YoloV5/data/images/bus.jpg', cv2.IMREAD_COLOR) # 读图片，忽略alpha通道
# b,g,r = cv2.split(img)
# img = cv2.merge([r,g,b])

# 原图
img = plt.imread('D:/User/Documents/Code/Encryption/Transfrom/ED_YoloV5/data/images/bus.jpg')
plt.imshow(img)
plt.show()

# 私钥
key = ProcessingKey(img)
keyd = [39, 16, 78, 6]
keyc = [39, 16, 77, 7]

# 加密图像
print('加密')
EncryImg = noColorEncry(img, key)
plt.imshow(EncryImg)
plt.title('encryption')
plt.show()

# 解密图像
print('解密')
DecryImg = noColorDecry(EncryImg, key)
plt.imshow(DecryImg)
plt.title('decryption')
plt.show()

# 比较原图与解密图的差异
gap = np.abs(DecryImg - img)
plt.imshow(gap)
plt.show()
print('差异', max(gap.flatten()))