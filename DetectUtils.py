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
def Overlap(xyxy):
    '''
    对检测框之间是否有相交的判断
    '''
    result = False
    # 检测是否有相交部分
    for obj_box in stack:
        x1 = np.max(obj_box[0], xyxy[0])
        x2 = np.min(obj_box[2], xyxy[2])
        y1 = np.max(obj_box[1], xyxy[1])
        y2 = np.min(obj_box[3], xyxy[3])

        area = (x2 - x1) * (y2 - y1)
        if area > 0:
            result = True
            break

    # if xyxy not in stack:
    stack.append(xyxy)

    return result

def OverlapEncryption(xyxy, mask, key, fuison_image, name = 'object'):
    '''
    提取像素加密后嵌入原图（先判断是否检测框是否有相交）
    '''
    # 
    pass

def DirectEncryption(xyxy, mask, key, fuison_image, name = 'object'):
    '''
    提取像素加密后嵌入原图（不判断是否检测框是否有相交）
    '''
    if mask != None:
        pass

        ## 选择mask加密区域，并转换为nx1x3维数组

        
    else:

        ## 选择区域
        roi = np.ascontiguousarray(fuison_image[int(xyxy[0]):int(xyxy[2]), int(xyxy[1]):int(xyxy[3]), :]) # w h c
        cv2.imshow(name, cv2whc(roi))
        cv2.waitKey(0)

        # 加密
        # key = ProcessingKey(roi)
        encryption_image = noColorEncry(roi, key)
        cv2.imshow(name+'encryption', cv2whc(encryption_image))
        cv2.waitKey(0)

        # 将加密图和原图融合
        fuison_image[int(xyxy[0]):int(xyxy[2]), int(xyxy[1]):int(xyxy[3]), :] = encryption_image.transpose((1,0,2))
        cv2.imshow(name+'roi encryption', cv2whc(fuison_image))
        cv2.waitKey(0)

        # 解密
        decryption_img = noColorDecry(encryption_image, key)
        cv2.imshow(name+'encryption', cv2whc(decryption_img))
        cv2.waitKey(0)

        fuison_image[int(xyxy[0]):int(xyxy[2]), int(xyxy[1]):int(xyxy[3]), :] = decryption_img
        cv2.imshow(name+'roi decryption', cv2whc(fuison_image))
        cv2.waitKey(0)
    
    return encryption_image, fuison_image

def RoIEcryption(image, key, label = None, type='object', is_overlap:bool = False):
    '''
    image：原图 key：私钥 label：加密的类别 type：加密方式
    '''
    ## 检测输入内容
    label = label if isinstance(label, list) else None if label == None else [label]
    type = type if type in ['object', 'segment'] else 'object'

    ## 返回检测结果
    model = initYOLOModel(type)
    result = runModel(model, cv2whc(image)) # 由于runModel是处理cv的，故这里转回cv
    
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
        if is_overlap and Overlap(xyxy):
            encryption_image, fuison_image = OverlapEncryption(xyxy, mask, key, fuison_image, name)
        else:
            encryption_image, fuison_image = DirectEncryption(xyxy, mask, key, fuison_image, name)

        encryption_object.append([encryption_image, xyxy, mask])

    return encryption_object, fuison_image

# ------------ YOLOv5 -----------------
import torch
import cv2

from utils.general import non_max_suppression, scale_boxes
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
    ## YOLOv5模型
    model = DetectMultiBackend(weights='Transfrom/ED_YoloV5/weights/yolov5x.pt',
                                data='./data/coco.yaml', device=device)
    return model

def runModel(model,
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
    mask = None
    ## 获取检测/分割结果
    for i, det in enumerate(pred): # batch size
        ### 获取掩码
        if type == 'segment':
            mask = process_mask(proto[i], det[:, 6:], det[:, :4], img.shape[2:], upsample=True)  # HWC
            masks = masks.permute(1, 2, 0).contiguous()
            masks = masks.cpu().numpy()
            masks = scale_image(masks.shape[:2], masks, image.shape)

        ### 将获取的包围盒缩放至图片大小
        det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], image.copy().shape).round()
        ### 获取每个对象的坐标等结果
        for j, (*xyxy, conf, cls) in enumerate(det):
            if mask != None:
                result.append([xyxy, conf, cls, mask[j]])
            else: result.append([xyxy, conf, cls, None])
    
    return result
