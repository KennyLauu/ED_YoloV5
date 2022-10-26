import time
import os
import sys
from pathlib import Path
import cv2
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from DetectUtils import text2key, cv2whc, SelectEnncryption, SelectAreaEncryption, SelectMaskEncryption

video_mask = []  # 用于保存目标检测获取的掩码
# label_id = None  # 获取到的物体track_id

def plot_bboxes(image, bboxes, mask, key, line_thickness=None, is_label=True):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    for (x1, y1, x2, y2, cls_id, pos_id, detect_id) in bboxes:
        if is_label == True:
            color = (((pos_id - 128) ** 2 + 64) % 256, (255 - pos_id * 16) % 256, (pos_id * 128 + pos_id ** 2) % 256)
            c1, c2 = (x1, y1), (x2, y2)
            cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(cls_id, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(image, '{}: {}'.format(pos_id, cls_id), (c1[0], c1[1] - 2), 0, tl / 3,
                        [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        else:
            if mask is not None:
                encryption_info, image = SelectMaskEncryption(cv2whc(image), cv2whc(mask[:, :, detect_id:detect_id + 1]), key)
            else:
                _, image = SelectAreaEncryption(cv2whc(image), [x1, y1, x2, y2], key)
            # _, image = SelectEnncryption(cv2whc(image), [x1, y1, x2, y2], cv2whc(mask[:, :, detect_id:detect_id + 1]), key)
            image = cv2whc(image)

    return image


def update_tracker(target_detector, image, frame_counter, is_label=True, label_id=None, deepsort=None):
    # new_faces = []

    start = time.time()
    _, bboxes, mask = target_detector.detect(image)
    print('yolo run spend: ', time.time() - start)
    if mask is not None:
        video_mask.append(mask)

    bbox_xywh = []
    confs = []
    clss = []

    for x1, y1, x2, y2, cls_id, conf in bboxes:
        obj = [
            int((x1 + x2) / 2), int((y1 + y2) / 2),
            x2 - x1, y2 - y1
        ]
        bbox_xywh.append(obj)
        confs.append(conf)
        clss.append(cls_id)

    xywhs = torch.Tensor(bbox_xywh)
    confss = torch.Tensor(confs)

    start = time.time()
    outputs = deepsort.update(xywhs, confss, clss, image, frame_counter)
    print('deep sort spend:', time.time() - start)

    bboxes2draw = []
    # face_bboxes = []
    # current_ids = []
    for value in list(outputs):
        x1, y1, x2, y2, cls_, track_id, current_detect_index = value
        if label_id is not None and track_id not in label_id:
            continue
        bboxes2draw.append(
            (x1, y1, x2, y2, cls_, track_id, current_detect_index)
        )

    image = plot_bboxes(image, bboxes2draw, mask, text2key('1,2,3,4'), is_label=is_label)

    return image, deepsort.tracker  # , new_faces, face_bboxes
