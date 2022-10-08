import torch
import numpy as np
import cv2
from PIL import Image
import platform

from utils.plots import colors
from utils.general import non_max_suppression, scale_boxes
from utils.dataloaders import LoadImages
from utils.segment.general import process_mask
from models.yolo import ClassificationModel, DetectionModel, SegmentationModel, AutoShape, DetectMultiBackend
from utils.plots import Annotator

device = torch.device('cuda')

# 选择模型，是分割还是识别
model = DetectMultiBackend(weights='D:/User/Documents/Github/yolov5/weights/yolov5x-seg.pt', device=device,
                           data='D:/User/Documents/Github/yolov5/data/coco128.yaml')
# model = DetectMultiBackend(weights='D:/User/Documents/Github/yolov5/weights/yolov5x-cls.pt',
#                            data='D:/User/Documents/Github/yolov5/data/coco.yaml')
# model = DetectMultiBackend(weights='Transfrom/ED_YoloV5/weights/yolov5x.pt',
                        #    data='D:/User/Documents/Github/yolov5/data/coco.yaml')

img_loc = ['D:/User/Documents/Code/Encryption/Transfrom/ED_YoloV5/data/images/zidane.jpg']  # img_loc 可以是列表，或者元组
dataset = LoadImages(img_loc)

# 执行
for path, img, org_img, _, _ in dataset:
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
    for i, det in enumerate(pred):  # det 为 [x1, y1, x2, y2, conf, cls, ...]
        # 设定锚框
        annotator = Annotator(org_img.copy(), line_width=1)
        # 获取掩码
        masks = process_mask(proto[i], det[:, 6:], det[:, :4], img.shape[2:], upsample=True)  # HWC
        # 将锚框进行缩放 化为图片上坐标
        det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], org_img.copy().shape).round()  # rescale boxes to im0 size
        annotator.masks(masks,
                        colors=[colors(x, True) for x in det[:, 5]])

        # 对其中的每个结果进行操作
        for j, (*xyxy, conf, cls) in enumerate(det[:, :6]):
            # 给物体标上框
            annotator.box_label(xyxy, model.names[int(cls)]+str(j), color=colors(int(cls), True))

        im0 = annotator.result()

        cv2.namedWindow('str(p)', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
        cv2.resizeWindow('str(p)', im0.shape[1], im0.shape[0])
        cv2.imshow('str(p)', im0)
        cv2.waitKey(0)
        # if cv2.waitKey(1) == ord('q'):  # 1 millisecond
        #     exit()

    # b = pred.numpy()
    # print(b[0, 0, 0, 0, 0]) # batch_size, xyxy



# img = cv2.imread(img_loc)[...,::-1] # opencv image (BGR to RGB)
# img = Image.open(img_loc)

# p = torch.empty(1, device=device)
# im = np.asarray(exif_transpose(img))
# im = im[..., :3] if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)  # enforce 3ch input
