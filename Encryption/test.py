# 加密/解密测试

# matplotlib==3.5.2
import time

import matplotlib.pyplot as plt
import numpy as np

from DetectUtils import cv2whc
from EncryUtils import ProcessingKey
from noColorDecry import noColorDecry
from noColorEncry import noColorEncry

#
# plt.axis('off')
# plt.xticks([])
# plt.yticks([])

# 原图
# plt.title('source image')
img = plt.imread('../data/images/bus.jpg')
# img = np.ascontiguousarray(img[:, :, :])
# plt.imshow(img)
# plt.show()
# img = cv2.imread('D:/User/Documents/Code/Encryption/Transfrom/ED_YoloV5/data/images/bus.jpg')
# img = np.ascontiguousarray(img[0:200, 0:200, :])
# cv2.imshow('title', img)
# cv2.waitKey(0)

# 私钥
key = ProcessingKey(img)
# print(key)
start = time.time()
# 加密图像
print('加密')
EncryImg = noColorEncry(img, key)
# plt.imshow(EncryImg.transpose(1, 0 ,2))
# plt.title('encryption')
# cv2.imshow('', cv2whc(EncryImg))
# cv2.waitKey(0)
# img = Image.fromarray(EncryImg)
# img.save('encryImg.png')

# EncryImg = cv2.imread('encryImg.png')

end = time.time()
print('加密所需时间 %f' % (end - start))

start = time.time()
# 解密图像
print('解密')
DecryImg = noColorDecry(cv2whc(EncryImg).transpose([1, 0, 2]), np.array([7, 19, 32, 106]))
# plt.imshow(DecryImg)
# plt.title('decryption')
# plt.axis('off')
# plt.xticks([])
# plt.yticks([])
# cv2.imshow('Decryption', cv2whc(DecryImg))
# cv2.waitKey(0)
end = time.time()
print('解密所需时间 %f' % (end - start))

# 比较原图与解密图的差异
# gap = np.abs(DecryImg - img)
# plt.imshow(gap)
# plt.show()
# print('差异', max(gap.flatten()))
