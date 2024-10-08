import os
import sys
from pathlib import Path

import cv2
import numpy as np
import time

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from DetectUtils import *

video_path = r'C:\Users\ltz\Desktop\srcimg\demo04.mp4'

cap = cv2.VideoCapture(video_path)
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame = 1

detect_type = 'segment'
key = np.array([1, 2, 3, 4])
label = 'person'
model = initYOLOModel(detect_type)

start = time.time()

video_save = cv2.VideoWriter('encryption_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (960, 540))

while (cap.isOpened()):
    ret, img = cap.read()

    if ret == False:
        break

    img = cv2.resize(img, (960, 540))
    # cv2.namedWindow('result', cv2.WINDOW_NORMAL)

    result, det = runModel(model, img, detect_type)
    fusion_image = cv2whc(img)

    # ------------
    # 全局操作
    stack.clear()
    # ------------

    # 对于每个检测出来的物体
    for i, obj in enumerate(result):
        # 解包内容
        xyxy, conf, cls, mask = obj

        if cls != 0.:
            continue

        # 重叠判定（若之前存在已加密的内容，则当前物体存在部分不需要加密）
        is_overlap, overlap_areas = Overlap(xyxy, mask)
        encryption_image, mask, fusion_image = OverlapEncryption(fusion_image, xyxy, key, overlap_areas, mask) \
            if is_overlap else \
            DirectEncryption(fusion_image, xyxy, key, mask)
    cv2.imshow('frame', cv2whc(fusion_image))

    video_save.write(cv2whc(fusion_image))

    print('fps: ', frame / (time.time() - start))
    frame += 1

    # 按下 q 退出播放
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_save.release()
cv2.destroyAllWindows()
