from unittest import result
import numpy as np
import cv2
from Encryption.EncryUtils import ProcessingKey
from Encryption.noColorEncry import noColorEncry
from Encryption.noColorDecry import noColorDecry
from utils.augmentations import letterbox

# ---------------- Utils ------------------
def cv2whc(img):
    '''
    将cv2的 `hwc bgr` 转换为 `whc rgb`
    或者将 `whc rgb` 转为 `hwc bgr`
    '''
    return np.ascontiguousarray(img.transpose((1, 0, 2))[:,:,::-1])

# ---------------- Encryption -------------
stack = []
def Overlap(xyxy, mask = None):
    '''
    对检测框之间是否有相交的判断
    mask：w x h x 1
    '''
    result = False
    overlap_areas = []
    x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
    # 检测是否有相交部分

    if mask != None:
        # 判断mask是否有可能有重叠部分
        for obj_area in stack:
            intersection = obj_area * mask 
            # 若存在相交，则intersection中值不全为0
            intersection_list = intersection[intersection == 1]

            if np.size(intersection_list) != 0:
                result = True
                overlap_areas.append(intersection)
    else: 
        # 只进行包围盒预测
        for obj_box in stack:
            x1 = torch.max(obj_box[0], xyxy[0])
            x2 = torch.min(obj_box[2], xyxy[2])
            y1 = torch.max(obj_box[1], xyxy[1])
            y2 = torch.min(obj_box[3], xyxy[3])

            if (x2 - x1) * (y2 - y1) > 0:
                result = True
                overlap_areas.append([x1, y1, x2, y2])

    # if xyxy not in stack:
    stack.append(xyxy)

    return result, overlap_areas

def OverlapEncryption(fuison_image, xyxy, key, overlap_areas, mask = None, name:str = 'object'):
    '''
    提取像素加密后嵌入原图（先判断是否检测框是否有相交）
    mask：全图的mask，若存在物体则标识为1，不存在物体标识为0；维度 w,h,1
    overlap_areas：存在相交部分，可以为boxes或者masks，但必须对应；存在相交为1，不存在为0
    '''

    w, h, c = fuison_image.shape

    overlap_mask = np.ones(shape = fuison_image.shape, dtype=np.uint8)

    ## 去除重叠部分
    if mask != None:

        assert overlap_area > 0, 'Tip: You should execute `DirectEncryption` function, nor this function!'
        assert mask.shape == overlap_areas[0].shape, 'The dimension of mask is not match the dimension of overlap_areas'

        overlap_mask = mask.repeat(c, axis = 2) # 将overlap_mask设置为目标mask
        ## 提取相交的部分，将其mask置为0（提取mask）
        for overlap_area in overlap_areas:
            overlap_area = overlap_area.repeat(c, axis = 2) # 扩展成 w,h,3
            overlap_mask[overlap_area == 1] = 0
    else:

        assert overlap_area > 0, 'Tip: You should execute `DirectEncryption` function, nor this function!'
        assert len(overlap_area[0]) == 4, 'Error dimension of boxes, please make sure get the boxes result `xyxy`'

        ## 提取相交的部分，将其mask置为0（提取包围盒）
        for overlap_box in overlap_areas:
            overlap_mask[int(overlap_box[0]):int(overlap_box[2]), int(overlap_box[1]):int(overlap_box[3]), :] = 0
    
    
    ## 选择区域
    roi = np.ascontiguousarray(fuison_image[int(xyxy[0]):int(xyxy[2]), int(xyxy[1]):int(xyxy[3]), :]) # w h c
    # cv2.imshow(name+'need encryption', cv2whc(roi))
    # cv2.waitKey(0)

    roi_mask = np.ascontiguousarray(overlap_mask[int(xyxy[0]):int(xyxy[2]), int(xyxy[1]):int(xyxy[3]), :])
    # cv2.imshow('roi_mask', cv2whc(roi_mask*255))
    # cv2.waitKey(0)

    ## 将区域相交的部分剔除
    encryption_list = roi[roi_mask == 1, np.newaxis] # n x 1 
    encryption_list = np.hstack((encryption_list[::3], encryption_list[1::3], encryption_list[2::3])).reshape(-1, 1, 3)
    
    ## ------------------------------------------------------------
    ## 错误处理 （当需要加密的区域与其他区域完全重叠，且被其他区域加密时）
    if np.size(encryption_list) == 0:
        return roi, roi_mask, fuison_image
    ## ------------------------------------------------------------

    ## 加密（将未相交的部分加密）
    encryption_result = noColorEncry(encryption_list, key)
    ## 还原（将未相交的加密部分还原）
    roi[roi_mask == 1, np.newaxis] = encryption_result.reshape(-1, 1)
    # cv2.imshow(name+'encryptioned roi', cv2whc(roi))
    # cv2.waitKey(0)

    ## 加密图像与原图融合
    fuison_image[int(xyxy[0]):int(xyxy[2]), int(xyxy[1]):int(xyxy[3]), :] = roi
    cv2.imshow(name + 'fuison image', cv2whc(fuison_image))
    cv2.waitKey(0)  

    encryption_image = roi
    mask = roi_mask

    ## 解密
    # decryption_list = encryption_image[mask == 1, np.newaxis]
    # decryption_list = np.hstack((decryption_list[::3], decryption_list[1::3], decryption_list[2::3])).reshape(-1, 1, 3)
    # decryption_result = noColorDecry(decryption_list, key)
    # encryption_image[mask == 1, np.newaxis] = decryption_result.reshape(-1, 1)
    # decryption_image = encryption_image
    # cv2.imshow(name+'decryption image', cv2whc(decryption_image))
    # cv2.waitKey(0)

    # fuison_image[int(xyxy[0]):int(xyxy[2]), int(xyxy[1]):int(xyxy[3]), :] = decryption_image
    # cv2.imshow(name+'roi decryption', cv2whc(fuison_image))
    # cv2.waitKey(0)

    return encryption_image, mask, fuison_image

def DirectEncryption(fuison_image, xyxy, key, mask = None, name:str = 'object'):
    '''
    提取像素加密后嵌入原图（不判断是否检测框是否有相交）
    对图像fuison_image的xyxy区域使用密钥key进行加密
    若包含mask，则可以传入mask，否则可以设为None
    '''
    if mask != None:
        ## 选择mask加密区域，并转换为nx1x3维数组
        ## 选择区域
        roi = np.ascontiguousarray(fuison_image[int(xyxy[0]):int(xyxy[2]), int(xyxy[1]):int(xyxy[3]), :]) # w h c
        # cv2.imshow(name+'need encryption', cv2whc(roi))
        # cv2.waitKey(0)

        roi_mask = np.ascontiguousarray(mask[int(xyxy[0]):int(xyxy[2]), int(xyxy[1]):int(xyxy[3]), :])
        # cv2.imshow('roi_mask', cv2whc(roi_mask*255))
        # cv2.waitKey(0)

        ## 将区域相交的部分剔除
        encryption_list = roi[roi_mask == 1, np.newaxis] # n x 1 
        encryption_list = np.hstack((encryption_list[::3], encryption_list[1::3], encryption_list[2::3])).reshape(-1, 1, 3)
        
        ## ------------------------------------------------------------
        ## 错误处理 （当需要加密的区域与其他区域完全重叠，且被其他区域加密时）
        if np.size(encryption_list) == 0:
            return roi, roi_mask, fuison_image
        ## ------------------------------------------------------------

        ## 加密（将未相交的部分加密）
        encryption_result = noColorEncry(encryption_list, key)
        ## 还原（将未相交的加密部分还原）
        roi[roi_mask == 1, np.newaxis] = encryption_result.reshape(-1, 1)
        # cv2.imshow(name+'encryptioned roi', cv2whc(roi))
        # cv2.waitKey(0)

        ## 加密图像与原图融合
        fuison_image[int(xyxy[0]):int(xyxy[2]), int(xyxy[1]):int(xyxy[3]), :] = roi
        cv2.imshow(name + 'fuison image', cv2whc(fuison_image))
        cv2.waitKey(0)  

        encryption_image = roi
        mask = roi_mask

        ## 解密
        # decryption_list = encryption_image[mask == 1, np.newaxis]
        # decryption_list = np.hstack((decryption_list[::3], decryption_list[1::3], decryption_list[2::3])).reshape(-1, 1, 3)
        # decryption_result = noColorDecry(decryption_list, key)
        # encryption_image[mask == 1, np.newaxis] = decryption_result.reshape(-1, 1)
        # decryption_image = encryption_image
        # cv2.imshow(name+'decryption image', cv2whc(decryption_image))
        # cv2.waitKey(0)

        # fuison_image[int(xyxy[0]):int(xyxy[2]), int(xyxy[1]):int(xyxy[3]), :] = decryption_image
        # cv2.imshow(name+'roi decryption', cv2whc(fuison_image))
        # cv2.waitKey(0)
        
    else:
        ## 选择区域
        roi = np.ascontiguousarray(fuison_image[int(xyxy[0]):int(xyxy[2]), int(xyxy[1]):int(xyxy[3]), :]) # w h c
        # cv2.imshow(name, cv2whc(roi))
        # cv2.waitKey(0)

        # 加密
        # key = ProcessingKey(roi)
        encryption_image = noColorEncry(roi, key)
        # cv2.imshow(name+'encryption', cv2whc(encryption_image))
        # cv2.waitKey(0)

        # 将加密图和原图融合
        fuison_image[int(xyxy[0]):int(xyxy[2]), int(xyxy[1]):int(xyxy[3]), :] = encryption_image.transpose((1,0,2))
        cv2.imshow(name+'roi encryption', cv2whc(fuison_image))
        cv2.waitKey(0)

        # 解密
        # decryption_image = noColorDecry(encryption_image, key)
        # cv2.imshow(name+'encryption', cv2whc(decryption_image))
        # cv2.waitKey(0)

        # fuison_image[int(xyxy[0]):int(xyxy[2]), int(xyxy[1]):int(xyxy[3]), :] = decryption_image
        # cv2.imshow(name+'roi decryption', cv2whc(fuison_image))
        # cv2.waitKey(0)
    
    return encryption_image, mask, fuison_image

def RoIEcryption(image, key, label:list|tuple = None, type='object'):
    '''
    image：原图 key：私钥 label：加密的类别 type：加密方式
    '''
    ## 检测输入内容
    label = label if isinstance(label, list|tuple) else None if label == None else [label]
    type = type if type in ['object', 'segment'] else 'object'

    ## 返回检测结果
    model = initYOLOModel(type)
    result = runModel(model, cv2whc(image), type) # 由于runModel是处理cv的，故这里转回cv
    
    # ------------
    # 全局操作
    stack.clear()
    # ------------

    encryption_object = []
    fuison_image = image

    ## 对于每个检测出来的物体
    for obj in iter(result):
        ## 解包内容
        xyxy, conf, cls, mask = obj
        name = model.names[int(cls)]
        ## 选择需要加密的物体
        if label != None and name not in label:
            continue

        ## 重叠判定（若之前存在已加密的内容，则当前物体存在部分不需要加密）
        is_overlap, overlap_areas = Overlap(xyxy, mask)
        encryption_image, mask, fuison_image = OverlapEncryption(fuison_image, xyxy, key, overlap_areas, mask, name) \
                                                if is_overlap else \
                                                DirectEncryption(fuison_image, xyxy, key, mask, name)

        ## 加密坐标
        # DirectEncryption(xyxy, None, key, xyxy)

        encryption_object.append([encryption_image, xyxy, mask])

    return encryption_object, fuison_image

# ------------ YOLOv5 -----------------
import torch
import torchvision
import cv2

from utils.general import scale_boxes, xywh2xyxy, non_max_suppression
from utils.dataloaders import LoadImages
from utils.segment.general import process_mask, scale_image
from models.yolo import ClassificationModel, DetectionModel, SegmentationModel, AutoShape, DetectMultiBackend
from utils.plots import Annotator, colors


def initYOLOModel(task:str = 'object'):
    '''
    初始化YOLO模型，返回YOLOv5模型  

    输入：
        task：选择**目标检测模型**或者**实例分割模型**，可以接受`object`或`segment`参数
    输出：    
        返回对应的YOLOv5模型
    '''
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if task == 'object':
        ## YOLOv5模型
        model = DetectMultiBackend(weights='Transfrom/ED_YoloV5/weights/yolov5x.pt',
                                    data='D:/User/Documents/Github/yolov5/data/coco.yaml', device=device)
    elif task == 'segment':
        model = DetectMultiBackend(weights='Transfrom/yolov5/weights/yolov5x-seg.pt', device=device,
                                   data='D:/User/Documents/Github/yolov5/data/coco128.yaml')
    else: Exception('task not exit ', task)
    return model

def runModel(model:ClassificationModel|DetectionModel|SegmentationModel|AutoShape|DetectMultiBackend, 
             image, type = 'object'):
    '''
    运行模型，返回执行结果

    输入：
        model：YOLO模型，可以通过 `initYOLOModel` 获取
        image：图片数组，可使用cv2或者plt获取对应的数组

    输出：
        返回对应图片检测的结果
    '''
    ## 图片处理
    img = letterbox(image)[0]  # padded resize
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)  # contiguous
    img = torch.from_numpy(img).to(model.device)
    img = img.half() if model.fp16 else img.float()  # uint8 to fp16/32
    img /= 255  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim

    ## 执行检测
    if type == 'object':
        pred = model(img)
        pred = non_max_suppression(pred)
    elif type == 'segment':
        pred, proto = model(img)[:2]
        pred = non_max_suppression(pred, nm = 32)
    else: Exception('type is error, only supposed `object` or `segment`')

    result = []
    masks = None
    ## 获取检测/分割结果
    for i, det in enumerate(pred): # batch size
        ### 获取掩码
        if type == 'segment':
            masks = process_mask(proto[i], det[:, 6:], det[:, :4], img.shape[2:], upsample=True)  # HWC
            masks = masks.permute(1, 2, 0).contiguous()
            masks = masks.cpu().numpy()
            masks = scale_image(masks.shape[:2], masks, image.shape)

        ### 将获取的包围盒缩放至图片大小
        det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], image.copy().shape).round()
        ### 获取每个对象的坐标等结果
        for j, (*xyxy, conf, cls) in enumerate(det):
            if masks != None:
                result.append([xyxy, conf, cls, masks[:, :, j:j+1]])
            else: result.append([xyxy, conf, cls, None])
    
    return result


# def non_max_suppression(
#         prediction,
#         conf_thres=0.25,
#         iou_thres=0.45,
#         classes=None,
#         agnostic=False,
#         multi_label=False,
#         labels=(),
#         max_det=300,
#         nm=0,  # number of masks
# ):
#     """
#     非极大值抑制，合并IoU较大的预测框，最大化加密
#     prediction的格式为 [batch_size, grid_cell, location+confidence+classes] e.g. 1,15120,
#     1. 过滤 存在物体的置信度 小于conf_thres的框
#     2. 过滤 目标置信度 小于conf_thres的框

#     """

#     if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
#         prediction = prediction[0]  # select only inference output

#     device = prediction.device
#     bs = prediction.shape[0]  # batch size
#     nc = prediction.shape[2] - nm - 5  # number of classes 类别个数
#     xc = prediction[..., 4] > conf_thres  # candidates 获取置信度大于conf_thres的检测框

#     # Checks
#     assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
#     assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

#     # Settings
#     # min_wh = 2  # (pixels) minimum box width and height
#     max_wh = 7680  # (pixels) maximum box width and height
#     max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
#     multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

#     mi = 5 + nc  # mask start index # mask的开始索引
#     output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
#     for xi, x in enumerate(prediction):  # image index, image inference
#         # Apply constraints
#         # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
#         x = x[xc[xi]]  # confidence

#         # Cat apriori labels if autolabelling
#         if labels and len(labels[xi]):
#             lb = labels[xi]
#             v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
#             v[:, :4] = lb[:, 1:5]  # box
#             v[:, 4] = 1.0  # conf
#             v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
#             x = torch.cat((x, v), 0)

#         # If none remain process next image
#         if not x.shape[0]:
#             continue

#         # Compute conf
#         x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

#         # Box/Mask
#         box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
#         mask = x[:, mi:]  # zero columns if no masks # 获取mask

#         # Detections matrix nx6 (xyxy, conf, cls)
#         if multi_label:
#             i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
#             x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
#         else:  # best class only
#             conf, j = x[:, 5:mi].max(1, keepdim=True)
#             x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

#         # Filter by class
#         if classes is not None:
#             x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

#         # Check shape
#         n = x.shape[0]  # number of boxes
#         if not n:  # no boxes
#             continue
#         elif n > max_nms:  # excess boxes
#             x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
#         else:
#             x = x[x[:, 4].argsort(descending=True)]  # sort by confidence 按置信度从高到低排序

#         # Batched NMS
#         c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
#         boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
#         i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS 非极大值抑制

#         # 判断框是否相交
#         # 若相交，则判断

#         if i.shape[0] > max_det:  # limit detections
#             i = i[:max_det]

#         output[xi] = x[i]

#     return output