import os
import sys
import time
from pathlib import Path

import cv2

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from DetectUtils import *

video_path = 'D:/User/Documents/Github/ED_YoloV5/data/videos/dog.mp4'

cap = cv2.VideoCapture(video_path)
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame = 1
fps = int(cap.get(5))

detect_type = 'segment'
key = np.array([1, 2, 3, 4])
label = 'person'
model = initYOLOModel(detect_type)

start = time.time()

videoWriter = None

while (cap.isOpened()):
    ret, img = cap.read()

    if ret == False:
        break

    # img = cv2.resize(img, (960, 540))
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

        if cls == 0:
            # 重叠判定（若之前存在已加密的内容，则当前物体存在部分不需要加密）
            is_overlap, overlap_areas = Overlap(xyxy, mask)
            encryption_image, mask, fusion_image = OverlapEncryption(fusion_image, xyxy, key, overlap_areas, mask) \
                if is_overlap else \
                DirectEncryption(fusion_image, xyxy, key, mask)

    print('fps: ', frame / (time.time() - start))
    frame += 1
    fusion_image = cv2whc(fusion_image)
    result = fusion_image.copy()
    fusion_image = cv2.resize(fusion_image, (960, 540))
    cv2.imshow('frame', fusion_image)

    if videoWriter is None:
        fourcc = cv2.VideoWriter_fourcc(
            'm', 'p', '4', 'v')  # opencv3.0
        videoWriter = cv2.VideoWriter(
            'result.mp4', fourcc, fps, (result.shape[1], result.shape[0]))
    videoWriter.write(result)

    # 按下 q 退出播放
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print('finish')
cap.release()
videoWriter.release()
cv2.destroyAllWindows()
