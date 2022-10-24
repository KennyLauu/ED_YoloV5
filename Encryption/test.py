# 加密/解密测试
import time
import matplotlib.pyplot as plt
import numpy as np

from EncryUtils import ProcessingKey
from noColorDecry import noColorDecry
from noColorEncry import noColorEncry

# 原图
img = plt.imread('../data/images/zidane.jpg')
img = np.ascontiguousarray(img[:, :, :])
plt.imshow(img)
plt.show()

# 私钥
key = ProcessingKey(img)

# 加密图像
print('加密')
start = time.time()
EncryImg = noColorEncry(img, key)
end = time.time()
print('加密花费', end - start)

plt.imshow(EncryImg)
plt.title('encryption')
plt.show()

# 解密图像
print('解密')
start = time.time()
DecryImg = noColorDecry(EncryImg, key)
end = time.time()
print('解密花费', end - start)

plt.imshow(DecryImg)
plt.title('decryption')
plt.show()

# 比较原图与解密图的差异
gap = np.abs(DecryImg - img)
# plt.imshow(gap)
# plt.show()
print('差异', max(gap.flatten()))
