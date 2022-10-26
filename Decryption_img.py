import os
import sys
import time
from pathlib import Path

from PIL import Image
import cv2

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from DetectUtils import *
from Encryption.EncryUtils import ProcessingKey

# img = r'C:\Users\ltz\Desktop\srcimg\person.jpg'
# encryption_method = 'seg'
# #
# img = Image.open(img)
# # cv2.imshow('plain image', img)
# # cv2.waitKey(0)
# # img = cv2whc(img)  # 将cv2的 hwc bgr 转为 whc rgb
# img = np.ascontiguousarray(img)
# key = ProcessingKey(img)
# print(key)
# img = PIL2whc(img)
# # label = ['person', 'horse'] # 加密的类别
# label = 'horse'
#
# # 返回原图，加密+原图等
# start = time.time()
# encryption_object, fusion_image = RoIEncryption(img, key, label, encryption_method)
# end = time.time()
# print('encryption + load YOLO spend: ', end - start)
# # cv2.imshow('encryption image', cv2whc(fusion_image))
# # cv2.waitKey(0)
# # cv2.imwrite('encryption_image.png', cv2whc(fusion_image))
# cv2.imwrite('segment_result_dog.png', cv2whc(fusion_image))
#
# # 解密内容
# # start = time.time()
# # plain_image = RoIDecryption(fusion_image, encryption_object, key)
# # end = time.time()
# # print('decryption image spend: ', end - start)
# # cv2.imshow('decryption image', cv2whc(plain_image))
# # cv2.waitKey(0)
# # cv2.imwrite('decryption_image.png', cv2whc(plain_image))
#
# # 差异
# # difference = np.array(img - plain_image)
# # print(np.max(difference.flatten()))


# 添加加密信息
# SetEncryptionImage('segment_result_dog.png', encryption_object, encryption_method, fusion_image, key)

# encryption_info = GetEncryptionImageInfo('encryption_image.png')

def Decryption_img(img, key):
    img = EncryptionImage2Decryption(img, key)
    return PIL2whc(img)

# img = Decryption_img(r'D:\Files\Chrome Download\Encrytion_image (20).png', '7, 19, 32, 106')
# cv2.imshow('a', img[:, :, ::-1])
# cv2.waitKey(0)