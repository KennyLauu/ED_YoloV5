import time

import cv2
import torch

from DetectUtils import text2key, cv2whc, SelectEnncryption, SelectAreaEncryption, SelectMaskEncryption
from deep_sort.deep_sort import DeepSort
from deep_sort.utils.parser import get_config

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
cfg = get_config()
cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)
video_mask = []  # 用于保存目标检测获取的掩码


def plot_bboxes(image, bboxes, mask, key, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    for (x1, y1, x2, y2, cls_id, pos_id, detect_id) in bboxes:
        color = (((pos_id - 128) ** 2 + 64) % 256, (255 - pos_id * 16) % 256, (pos_id * 128 + pos_id ** 2) % 256)
        c1, c2 = (x1, y1), (x2, y2)
        cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(cls_id, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, '{}: {}'.format(pos_id, cls_id), (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        # _, image = SelectAreaEncryption(image, [y1, x1, y2, x2], key)
        encryption_info, image = SelectMaskEncryption(cv2whc(image), cv2whc(mask[:, :, detect_id:detect_id + 1]), key)
        # _, image = SelectEnncryption(cv2whc(image), [x1, y1, x2, y2], cv2whc(mask[:, :, detect_id:detect_id + 1]), key)

        image = cv2whc(image)

    return image


def update_tracker(target_detector, image, frame_counter):
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
    label_id = [1]  # 获取到的物体track_id
    # face_bboxes = []
    # current_ids = []
    for value in list(outputs):
        x1, y1, x2, y2, cls_, track_id, current_detect_index = value
        if label_id is not None and track_id not in label_id:
            continue
        bboxes2draw.append(
            (x1, y1, x2, y2, cls_, track_id, current_detect_index)
        )
        # current_ids.append(track_id)
        # if cls_ == 'face':
        #     if not track_id in target_detector.faceTracker:
        #         target_detector.faceTracker[track_id] = 0
        #         face = image[y1:y2, x1:x2]
        #         new_faces.append((face, track_id))
        #     face_bboxes.append(
        #         (x1, y1, x2, y2)
        #     )

    # ids2delete = []
    # for history_id in target_detector.faceTracker:
    #     if not history_id in current_ids:
    #         target_detector.faceTracker[history_id] -= 1
    #     if target_detector.faceTracker[history_id] < -5:
    #         ids2delete.append(history_id)
    #
    # for ids in ids2delete:
    #     target_detector.faceTracker.pop(ids)
    #     print('-[INFO] Delete track id:', ids)

    image = plot_bboxes(image, bboxes2draw, mask, text2key('1,2,3,4'))

    return image, deepsort.tracker  # , new_faces, face_bboxes
