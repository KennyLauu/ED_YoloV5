import torch
import numpy as np
import cv2
from PIL import Image
import platform

from utils.plots import colors
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox
from utils.dataloaders import LoadImages
from utils.segment.general import process_mask, scale_image
from models.yolo import ClassificationModel, DetectionModel, SegmentationModel, AutoShape, DetectMultiBackend
from utils.plots import Annotator

device = torch.device('cuda')

# 选择模型，是分割还是识别
model = DetectMultiBackend(weights='D:/User/Documents/Github/yolov5/weights/yolov5x-seg.pt', device=device,
                           data='D:/User/Documents/Github/yolov5/data/coco128.yaml')
# model = DetectMultiBackend(weights='D:/User/Documents/Github/yolov5/weights/yolov5x-cls.pt',
#                            data='D:/User/Documents/Github/yolov5/data/coco.yaml')
# model = DetectMultiBackend(weights='Transfrom/ED_YoloV5/weights/yolov5x.pt',
#                            data='D:/User/Documents/Github/yolov5/data/coco.yaml')

# img_loc = './Transfrom/ED_YoloV5/data/images/zidane.jpg'  # img_loc 可以是列表，或者元组
img_loc = ['D:/User/Documents/Code/Encryption/Transfrom/ED_YoloV5/data/images/zidane.jpg',
            'D:/User/Documents/Code/Encryption/ROI chaotic image encryption based on lifting scheme and YOLOv5/images/person.jpg']  # img_loc 可以是列表，或者元组
# dataset = LoadImages(img_loc)
imgs = []
org_img = []
for img_l in img_loc:
    img_a = cv2.imread(img_l)
    org_img.append(img_a)
    img_a = letterbox(img_a)[0]  # padded resize
    img_a = img_a.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img_a = np.ascontiguousarray(img_a)  # contiguous
    imgs.append(img_a)

result = []
# 执行 (这里为什么有个循环呢？因为维度不匹配，图片宽高各异，实在没办法同一放在一个np中)
for img_index, img in enumerate(imgs):
    img = torch.from_numpy(img).to(model.device)
    img = img.half() if model.fp16 else img.float()  # uint8 to fp16/32
    img /= 255  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim

    # for segment
    pred, proto = model(img)[:2]
    pred = non_max_suppression(pred, nm = 32)
    # for detection
    # pred = model(img)
    # pred = non_max_suppression(pred)

    # 多个预测结果
    # for i, det in enumerate(pred):  # det 为 [x1, y1, x2, y2, conf, cls, ...]
    det = pred[0]
    i = 0
    # 设定锚框
    annotator = Annotator(org_img[img_index].copy(), line_width=1)
    # 获取掩码
    masks = process_mask(proto[i], det[:, 6:], det[:, :4], img.shape[2:], upsample=True)  # HWC
    # 将锚框进行缩放 化为图片上坐标
    det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], org_img[img_index].copy().shape).round()  # rescale boxes to im0 size
    annotator.masks(masks,
                    colors=[colors(x, True) for x in det[:, 5]])
    masks = masks.permute(1, 2, 0).contiguous()
    masks = masks.cpu().numpy()
    masks = scale_image(masks.shape[:2], masks, org_img[img_index].shape)
    masks *= 255
    cv2.imshow('masks', np.ascontiguousarray(masks[img_index]))
    cv2.waitKey(0)

    # 对其中的每个结果进行操作
    for j, (*xyxy, conf, cls) in enumerate(det[:, :6]):
        # 给物体标上框
        annotator.box_label(xyxy, model.names[int(cls)]+str(j), color=colors(int(cls), True))
        result.append([xyxy, conf, cls, masks[j]])

    im0 = annotator.result()

    cv2.namedWindow('str(p)', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
    cv2.resizeWindow('str(p)', im0.shape[1], im0.shape[0])
    cv2.imshow('str(p)', im0)
    cv2.waitKey(0)
    # if cv2.waitKey(1) == ord('q'):  # 1 millisecond
    #     exit()

for obj in iter(result):
    # pass
    xyxy, conf, cls, mask = obj
    name = model.names[int(cls)]
    print('坐标', xyxy[0], xyxy[1], xyxy[2], xyxy[3])
    print('置信度', conf)
    print('类别', name)
