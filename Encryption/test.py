# 加密/解密测试

import numpy as np
import cv2
import matplotlib.pyplot as plt
# matplotlib==3.5.2

from noColorDecry import noColorDecry
from noColorEncry import noColorEncry
from EncryUtils import ProcessingKey

# img = cv2.imread('D:/User/Documents/Code/Encryption/Transfrom/ED_YoloV5/data/images/bus.jpg', cv2.IMREAD_COLOR) # 读图片，忽略alpha通道
# b,g,r = cv2.split(img)
# img = cv2.merge([r,g,b])

# 原图
img = plt.imread('../data/images/bus.jpg')
img = np.ascontiguousarray(img[0:223, 0:223, :])
plt.imshow(img)
plt.show()
# img = cv2.imread('D:/User/Documents/Code/Encryption/Transfrom/ED_YoloV5/data/images/bus.jpg')
# img = np.ascontiguousarray(img[0:200, 0:200, :])
# cv2.imshow('title', img)
# cv2.waitKey(0)

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