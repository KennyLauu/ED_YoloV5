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

img = './data/images/person.jpg'

img = cv2.imread(img)
cv2.imshow('plain image', img)
cv2.waitKey(0)
img = cv2whc(img) # 将cv2的 hwc bgr 转为 whc rgb

key = ProcessingKey(img)

label = ['person', 'horse'] # 加密的类别
# 返回原图，加密+原图等
result = RoIEcryption(img, key)

# plt.imshow(img)
# plt.show()