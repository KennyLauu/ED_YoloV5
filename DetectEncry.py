import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
from pathlib import Path
import os

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from DetectUtils import RoIEcryption, cv2whc
from Encryption.EncryUtils import ProcessingKey

# def DetectEncryption():

img = 'D:/User/Documents/Code/Encryption/ROI chaotic image encryption based on lifting scheme and YOLOv5/images/person.jpg'
# img = 'D:/User/Documents/Code/Encryption/Transfrom/ED_YoloV5/data/images/zidane.jpg'

img = cv2.imread(img)
cv2.imshow('plain image', img)
cv2.waitKey(0)
img = cv2whc(img) # 将cv2的 hwc bgr 转为 whc rgb

key = ProcessingKey(img)

# label = ['person', 'horse'] # 加密的类别
label = None
# 返回原图，加密+原图等
# encryption_object, fuison_image = RoIEcryption(img, key, label, type='segment')
encryption_object, fuison_image = RoIEcryption(img, key, label, type='object')

cv2.imshow('encryption image' ,cv2whc(fuison_image))
cv2.waitKey(0)