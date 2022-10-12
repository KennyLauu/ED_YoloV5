import os
import sys
from pathlib import Path

import cv2
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from DetectUtils import RoIEcryption, cv2whc, RoIDecryption
from Encryption.EncryUtils import ProcessingKey

# def DetectEncryption():

img = 'D:/User/Pictures/学习/深度学习/微信图片_20221010172412.jpg'
# img = 'D:/User/Documents/Code/Encryption/ROI chaotic image encryption based on lifting scheme and YOLOv5/images/dog.jpg'
# img = 'D:/User/Documents/Code/Encryption/ROI chaotic image encryption based on lifting scheme and YOLOv5/images/person.jpg'
# img = './data/images/zidane.jpg'
# img = './data/images/bus.jpg'

img = cv2.imread(img)
# cv2.imshow('plain image', img)
# cv2.waitKey(0)
img = cv2whc(img)  # 将cv2的 hwc bgr 转为 whc rgb

key = ProcessingKey(img)

# label = ['person', 'horse'] # 加密的类别
label = None

# 返回原图，加密+原图等
encryption_object, fusion_image = RoIEcryption(img, key, label, detect_type='segment')
# encryption_object, fusion_image = RoIEcryption(img, key, label, type='object')
cv2.imshow('encryption image', cv2whc(fusion_image))
cv2.waitKey(0)
# cv2.imwrite('segment_result_dog.jpg', cv2whc(fusion_image))

# 解密内容
plain_image = RoIDecryption(fusion_image, encryption_object, key)
cv2.imshow('decryption image', cv2whc(plain_image))
cv2.waitKey(0)

# 差异
difference = np.array(img - plain_image)
print(np.max(difference.flatten()))
