import numpy as np

from Encryption.noColorDecry import noColorDecry
from Encryption.noColorEncry import noColorEncry
from utils.augmentations import letterbox


# ---------------- Utils ------------------
def cv2whc(img):
    """
    将cv2的 `hwc bgr` 转换为 `whc rgb`
    或者将 `whc rgb` 转为 `hwc bgr`
    """
    return np.ascontiguousarray(img.transpose((1, 0, 2))[:, :, ::-1])

def text2key(text):
    """
    需要传入解密密钥，格式为 'a,b,c,d' 字符串或者列表 [a,b,c,d]
    """
    text = str.split(text, ',') if isinstance(text, str) else text 

    assert isinstance(text, list), 'error text type, must be str or list'
    assert len(text) == 4, 'error dimension, must be 4-dimension, shape=(4,)'

    # 将字符转换为数字
    text = np.array([int(val) for val in text])
    return text

def bytes2string(data:bytes):
    return data.decode('ascii')

def bytes2int(data:bytes):
    return int(bytes2string(data))

def get_wh(data:bytes):
    w_h_list = data.split(b',')
    return bytes2int(w_h_list[0]), bytes2int(w_h_list[1])

def get_xyxy(data:bytes):
    xyxy_list = data.split(b',')
    return [bytes2int(xyxy_list[0]),
            bytes2int(xyxy_list[1]),
            bytes2int(xyxy_list[2]),
            bytes2int(xyxy_list[3])]

def bytes2numpy(data:bytes):
    return np.frombuffer(data, dtype=np.uint8)

def get_object(data:bytes):
    object_list = data.split(b'],')
    encryption_object = []
    for obj in object_list[:-1]:
        combine_list = obj.split(b'|')
        xyxy = get_xyxy(combine_list[0][1:])
        if combine_list[1] != b' ':
            mask = bytes2numpy(combine_list[1])
        else: mask = None
        encryption_object.append([xyxy, mask])
    return encryption_object

_unpack_function_ = {b'method':bytes2string, b'shape':get_wh,
                   b'object_number':bytes2int, b'object':get_object}

def SetEncryptionImage(image_path, encryption_object, encryption_method:str = 'object', image_array:np = None):
    """
    Write Some Tip for Encryption Image
    """
    w,h,c = 0,0,0
    if image_array is not None:
        w,h,c = image_array.shape

    is_encryption = False

    with open(image_path, mode='rb+') as binary_img:
        binary_img.seek(-14, 2) # 获取最后四个字节

        encryption_info_aready = binary_img.readlines()[-1]
        
        # 已加密则取消继续加密
        if b'encryption_end' in encryption_info_aready:
            is_encryption = True

    if is_encryption == False:
        # 加密的文件头，包含加密方式，图片宽高，加密物体的数量
        with open(image_path, mode='ab') as binary_img:
            encryption_info = 'method:{};shape:{},{};object_number:{};object:'.format(
                encryption_method, w, h, len(encryption_object)
            )
            binary_img.write(encryption_info.encode('ascii'))

            # 添加每个物体
            for _ ,xyxy, mask in encryption_object:
                encryption_object_info = '[{},{},{},{}|'.format(
                    int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                )
                encryption_object_info_bytes = encryption_object_info.encode('ascii')
                encryption_object_info_bytes += mask[:,:,0:1].tobytes() if mask is not None else b' '
                encryption_object_info_bytes += b'],'
                binary_img.write(encryption_object_info_bytes)

            encryption_info = ';'
            encryption_info += 'encryption_end'
            binary_img.write(encryption_info.encode('ascii'))
    

def GetEncryptionImageInfo(image_path):
    with open(image_path, mode='rb') as binary_img:
        # 获取最后一行
        encryption_info_bytes = binary_img.read()
        encryption_info_start_index = encryption_info_bytes.find(b'method:')
        encryption_info_bytes = encryption_info_bytes[encryption_info_start_index:]

        # 获取需要的内容
        encryption_info_split_list = encryption_info_bytes.split(b';')

        encryption_info = {}

        for info in encryption_info_split_list:
            data_list = info.split(b':')
            if len(data_list) == 2:
                command, data = data_list
                encryption_info[bytes2string(command)] = _unpack_function_[command](data)
                print(bytes2string(command), ': ', encryption_info[bytes2string(command)], '\n')

        return encryption_info


def EncryptionImage2Decryption(image_path, key):
    # 将字符串key转为np
    key = text2key(key) if isinstance(key, str) else key
    # 获取图片中的掩码
    encryption_info = GetEncryptionImageInfo(image_path)
    encryption_objects = encryption_info['object']

    import cv2
    # 使用cv2读取图片
    fusion_image = cv2whc(cv2.imread(image_path))
    img_w, img_h, c = fusion_image.shape 

    encryption_object = []
    
    for obj in encryption_objects:
        [xyxy, mask] = obj
        w = xyxy[2] - xyxy[0]
        h = xyxy[3] - xyxy[1]
        encryption_image = fusion_image[int(xyxy[0]):int(xyxy[2]), int(xyxy[1]):int(xyxy[3]), :]
        if mask is not None:
            assert w*h == len(mask) or w*h*c == len(mask), 'error dimension {}x{} to mask dimension {}'.format(w, h, len(mask))
            roi_mask = mask.reshape(w, h, -1)
            if roi_mask.shape[2] != c:
                roi_mask = roi_mask.repeat(c, axis=2)  # 扩充mask w,h,c
        else: roi_mask = None
        encryption_object.append([encryption_image, xyxy, roi_mask])
        
    
    return RoIDecryption(fusion_image, encryption_object, key)

# ---------------- Encryption -------------
# stack 用于预测框的相交检测，请在调用前使用stack.clear()清空之前的值
stack = []


def Overlap(xyxy, mask=None):
    """
    对预测框之间是否有相交的判断，可以对mask或者boxes进行检测，返回是否相交的结果，
    若result为false，则overlap_area列表为空
    Detect intersection of prediction boxes
    Input:
        xyxy: the prediction boxes location left-top and right-bottom, need shape=(4,)
        mask: the mask contain whole images w x h x 1, and 1 if contain object else 0
    Output:
        result: whether there are intersection result, True or False
        overlap_area: the list array, contain mask or boxes(left-top, right-bottom) list of intersection.
    """
    result = False
    overlap_areas = []
    # 检测是否有相交部分

    if mask is not None:
        # 判断mask是否有可能有重叠部分
        for obj_area in stack:
            intersection = obj_area * mask
            # 若存在相交，则intersection中值不全为0
            intersection_list = intersection[intersection == 1]

            if np.size(intersection_list) != 0:
                result = True
                overlap_areas.append(intersection)

        stack.append(mask)
    else:
        # 只进行包围盒预测
        for obj_box in stack:
            x1 = max(obj_box[0], xyxy[0])
            x2 = min(obj_box[2], xyxy[2])
            y1 = max(obj_box[1], xyxy[1])
            y2 = min(obj_box[3], xyxy[3])

            if (x2 - x1) * (y2 - y1) > 0:
                result = True
                overlap_areas.append([x1, y1, x2, y2])

        stack.append(xyxy)

    return result, overlap_areas


def OverlapEncryption(fusion_image, xyxy, key, overlap_areas, mask=None, name: str = 'object'):
    """
    提取像素加密后嵌入原图（先判断是否检测框是否有相交）
    mask：全图的mask，若存在物体则标识为1，不存在物体标识为0；维度 w,h,1
    overlap_areas：存在相交部分，可以为boxes或者masks，但必须对应；存在相交为1，不存在为0

    Encrypting the `fusion image` in `xyxy` area with `key`, because there are `overlap areas`,
    we remove the intersection area, and there are better if you have `mask`, we can use
    mask for segment encryption or remove intersection area more accurately.

    Input:
        fusion image: plain image or image with portion encryption
        xyxy: the boxes (left-top, right-bottom)
        key: the secret key for encryption, generator by `ProcessingKey` function
        overlap area: the overlap, if there are no overlap areas, you should use `DirectEncryption` function
        mask: for instant segment, or you could set it None, but just remember use xyxy
        name: preserve for test

    Output:
        encryption image: just for preview, not use for decryption. (but actually is ok)
        mask: the area of encryption, use it to decryption.
        fusion image: for decryption, it contains the whole image.
    """

    w, h, c = fusion_image.shape

    overlap_mask = np.ones(shape=fusion_image.shape, dtype=np.uint8)

    # 去除mask中的重叠部分
    if mask is not None:

        assert len(overlap_areas) > 0, 'Tip: You should execute `DirectEncryption` function, nor this function!'
        assert mask.shape == overlap_areas[0].shape, 'The dimension of mask is not match the dimension of overlap_areas'

        overlap_mask = mask.repeat(c, axis=2)  # 将overlap_mask设置为目标mask
        # 提取相交的部分，将其mask置为0（提取mask）
        for overlap_area in overlap_areas:
            overlap_area = overlap_area.repeat(c, axis=2)  # 扩展成 w,h,3
            overlap_mask[overlap_area == 1] = 0
    else:

        assert len(overlap_areas) > 0, 'Tip: You should execute `DirectEncryption` function, nor this function!'
        assert len(overlap_areas[0]) == 4, 'Error dimension of boxes, please make sure get the boxes result `xyxy`'

        # 提取相交的部分，将其mask置为0（提取包围盒）
        for overlap_box in overlap_areas:
            overlap_mask[int(overlap_box[0]):int(overlap_box[2]), int(overlap_box[1]):int(overlap_box[3]), :] = 0

    return DirectEncryption(fusion_image, xyxy, key, overlap_mask, name)


def DirectEncryption(fusion_image, xyxy, key, mask=None, name: str = 'object'):
    """
    提取像素加密后嵌入原图（不判断是否检测框是否有相交）
    对图像fusion_image的xyxy区域使用密钥key进行加密
    若包含mask，则可以传入mask，否则可以设为None，mask维度为原始图片(w,h,?)

    Encrypt the `fusion image` in `xyxy` area with `key`, and there are better if you have `mask`
    we could encrypt image more accurately.

    Input:
        fusion image: plain image or image with portion encryption
        xyxy: the boxes (left-top, right-bottom)
        key: the secret key for encryption, generator by `ProcessingKey` function
        mask: for instant segment, or you could set it None, but just remember set xyxy
        name: preserve for test

    Output:
        encryption image: just for preview, not use for decryption. (but actually is ok)
        mask: the area of encryption, use it to decryption.
        fusion image: for decryption, it contains the whole image.
    """

    w, h, c = fusion_image.shape

    if mask is not None:
        # 选择mask加密区域，并转换为nx1x3维数组
        if np.ndim(mask) != 3:
            mask = np.expand_dims(mask, axis=2)  # w,h,1
        if mask.shape[2] != c:
            mask = mask.repeat(c, axis=2)  # 扩充mask w,h,c

        # 选择区域
        roi = np.ascontiguousarray(fusion_image[int(xyxy[0]):int(xyxy[2]), int(xyxy[1]):int(xyxy[3]), :])  # w h c
        # cv2.imshow(name+'need encryption', cv2whc(roi))
        # cv2.waitKey(0)

        roi_mask = np.ascontiguousarray(mask[int(xyxy[0]):int(xyxy[2]), int(xyxy[1]):int(xyxy[3]), :])
        # cv2.imshow('roi_mask', cv2whc(roi_mask))
        # cv2.waitKey(0)

        # 将区域相交的部分剔除
        encryption_list = roi[roi_mask == 1, np.newaxis]  # n x 1
        encryption_list = np.hstack((encryption_list[::3],
                                     encryption_list[1::3],
                                     encryption_list[2::3])).reshape(-1, 1, 3)

        # ------------------------------------------------------------
        # 错误处理 （当需要加密的区域与其他区域完全重叠，且被其他区域加密时）
        if np.size(encryption_list) <= 4:
            return roi, roi_mask, fusion_image
        # ------------------------------------------------------------

        # 加密（将未相交的部分加密）
        encryption_result = noColorEncry(encryption_list, key)
        # 还原（将未相交的加密部分还原）
        roi[roi_mask == 1, np.newaxis] = encryption_result.reshape(-1, 1)
        # cv2.imshow(name+'encryption roi', cv2whc(roi))
        # cv2.waitKey(0)

        # 加密图像与原图融合
        fusion_image[int(xyxy[0]):int(xyxy[2]), int(xyxy[1]):int(xyxy[3]), :] = roi
        # cv2.imshow(name + 'fusion image', cv2whc(fusion_image))
        # cv2.waitKey(0)  

        encryption_image = roi * roi_mask
        mask = roi_mask

        # 解密
        # decryption_list = encryption_image[mask == 1, np.newaxis]
        # decryption_list = np.hstack((decryption_list[::3],
        #                              decryption_list[1::3],
        #                              decryption_list[2::3])).reshape(-1, 1, 3)
        # decryption_result = noColorDecry(decryption_list, key)
        # encryption_image[mask == 1, np.newaxis] = decryption_result.reshape(-1, 1)
        # decryption_image = encryption_image
        # cv2.imshow(name+'decryption image', cv2whc(decryption_image))
        # cv2.waitKey(0)

        # fusion_image[int(xyxy[0]):int(xyxy[2]), int(xyxy[1]):int(xyxy[3]), :] = decryption_image
        # cv2.imshow(name+'roi decryption', cv2whc(fusion_image))
        # cv2.waitKey(0)

    else:
        # 选择区域
        roi = np.ascontiguousarray(fusion_image[int(xyxy[0]):int(xyxy[2]), int(xyxy[1]):int(xyxy[3]), :])  # w h c
        # cv2.imshow(name, cv2whc(roi))
        # cv2.waitKey(0)

        # 加密
        encryption_image = noColorEncry(roi, key)
        # cv2.imshow(name+'encryption', cv2whc(encryption_image))
        # cv2.waitKey(0)

        # 将加密图和原图融合
        fusion_image[int(xyxy[0]):int(xyxy[2]), int(xyxy[1]):int(xyxy[3]), :] = encryption_image
        # cv2.imshow(name+'roi encryption', cv2whc(fusion_image))
        # cv2.waitKey(0)

        # 创建mask
        # mask = np.zeros(fusion_image.shape)
        # mask[int(xyxy[0]):int(xyxy[2]), int(xyxy[1]):int(xyxy[3]), :] = 1

        # 解密
        # decryption_image = noColorDecry(encryption_image, key)
        # cv2.imshow(name+'encryption', cv2whc(decryption_image))
        # cv2.waitKey(0)

        # fusion_image[int(xyxy[0]):int(xyxy[2]), int(xyxy[1]):int(xyxy[3]), :] = decryption_image
        # cv2.imshow(name+'roi decryption', cv2whc(fusion_image))
        # cv2.waitKey(0)

    return encryption_image, mask, fusion_image


def RoIEncryption(image, key, label: list = None, detect_type='object'):
    """
    image：原图 key：私钥 label：加密的类别 detect_type：加密方式

    Encrypt image with automatic recognize. You need pass the plain image, and with key to
    encrypt, or you can choose to encrypt some certain label object, we can detect it and
    encrypt. You can choose the type for object detection use prediction boxes or instant
    segment for mask.

    Input:
        image: the plain image, numpy array
        key: the secret key, use it to encrypt image
        label: you can pass the label you need encryption
        detect type: the yolo for object detection or instant segment

    Output:
        **encryption object**: the encryption result contains portion encryption image, location
        and mask
        **fusion image**: the whole encryption image
    """
    # 检测输入内容
    label = label if type(label) in [list, tuple] else None if label is None else [label]
    detect_type = detect_type if detect_type in ['object', 'segment'] else 'object'

    import time
    start = time.time()
    # 返回检测结果
    model = initYOLOModel(detect_type)
    result, _ = runModel(model, cv2whc(image), detect_type)  # 由于runModel是处理cv的，故这里转回cv
    end = time.time()
    print('run yolo spend: ', end - start)

    # ------------
    # 全局操作
    stack.clear()
    # ------------

    encryption_object = []
    fusion_image = image

    # 对于每个检测出来的物体
    for obj in iter(result):
        # 解包内容
        xyxy, conf, cls, mask = obj
        name = model.names[int(cls)]
        # 选择需要加密的物体
        if label is not None and name not in label:
            continue

        # 重叠判定（若之前存在已加密的内容，则当前物体存在部分不需要加密）
        is_overlap, overlap_areas = Overlap(xyxy, mask)
        encryption_image, mask, fusion_image = OverlapEncryption(fusion_image, xyxy, key, overlap_areas, mask, name) \
            if is_overlap else \
            DirectEncryption(fusion_image, xyxy, key, mask, name)

        # 加密坐标
        # pos_list = np.array(xyxy, ndmin=3).reshape(-1, 1, 1)  # 4x1x1
        # print('encryption position before: ', pos_list)
        # encryption_position = noColorEncry(pos_list, key)
        # print('encryption position after: ', encryption_position)

        encryption_object.append([encryption_image, xyxy, mask])

    return encryption_object, fusion_image


def SelectAreaEcryption(image, xyxy, key):
    assert len(xyxy) == 4, 'please make sure the xyxy is 4-dimension, xyxy必须是4维的'
    assert xyxy[0] < xyxy[2] and xyxy[1] < xyxy[3], 'box position is error, must be left-top and right-bottom'

    fusion_image = image
    encryption_object = []

    # 重叠判定（若之前存在已加密的内容，则当前物体存在部分不需要加密）
    is_overlap, overlap_areas = Overlap(xyxy)
    encryption_image, mask, fusion_image = OverlapEncryption(fusion_image, xyxy, key, overlap_areas) \
        if is_overlap else \
        DirectEncryption(fusion_image, xyxy, key)

    # 加密坐标
    # pos_list = np.array(xyxy, ndmin=3).reshape(-1, 1, 1)  # 4x1x1
    # print('encryption position before: ', pos_list)
    # encryption_position = noColorEncry(pos_list, key)
    # print('encryption position after: ', encryption_position)

    encryption_object.append([encryption_image, xyxy, mask])

    return encryption_object, fusion_image


# ------------ Decryption -----------------

def RoIDecryption(fusion_image, encryption_object, key):
    """
    To decrypt image which encrypted by RoIEncryption function, you should upload
    the encrypted image, and the encryption_object that contain encryption_image,
    xyxy, mask and key.
    Args:
        fusion_image: image with encryption, whole encrypt image or local encrypt image.
        encryption_object: contain encryption image, xyxy, mask and key.
        key: the secret key for decryption.

    Returns: decryption image

    """

    w, h, c = fusion_image.shape

    for encry_obj in encryption_object:
        _, xyxy, mask = encry_obj

        # 解密坐标
        # decryption_position = noColorDecry(encryption_position, key)
        # xyxy = [val for val in decryption_position]
        # print('decryption position', xyxy)

        encryption_image = fusion_image[int(xyxy[0]):int(xyxy[2]), int(xyxy[1]):int(xyxy[3]), :]

        if mask is not None:
            # 选择mask加密区域，并转换为nx1x3维数组
            if np.ndim(mask) != 3:
                mask = np.expand_dims(mask, axis=2)  # w,h,1
            if mask.shape[2] != c:
                mask = mask.repeat(c, axis=2)  # 扩充mask w,h,c

            decryption_list = encryption_image[mask == 1, np.newaxis]  # nx1x3
            decryption_list = np.hstack((decryption_list[::3],
                                         decryption_list[1::3],
                                         decryption_list[2::3])).reshape(-1, 1, 3)

            # --------------------------------------------------------------------
            # 错误处理（不需要解密，由于与其他物体重叠较大，当其他物体解密后，该物体也会被解密）
            if np.size(decryption_list) <= 4:
                continue
            # --------------------------------------------------------------------

            decryption_result = noColorDecry(decryption_list, key)
            encryption_image[mask == 1, np.newaxis] = decryption_result.reshape(-1, 1)
            decryption_image = encryption_image
            # cv2.imshow('decryption image', cv2whc(decryption_image))
            # cv2.waitKey(0)

            fusion_image[int(xyxy[0]):int(xyxy[2]), int(xyxy[1]):int(xyxy[3]), :] = decryption_image
            # cv2.imshow('roi decryption', cv2whc(fusion_image))
            # cv2.waitKey(0)
        else:

            decryption_image = noColorDecry(encryption_image, key)
            # cv2.imshow('encryption', cv2whc(decryption_image))
            # cv2.waitKey(0)

            fusion_image[int(xyxy[0]):int(xyxy[2]), int(xyxy[1]):int(xyxy[3]), :] = decryption_image
            # cv2.imshow('roi decryption', cv2whc(fusion_image))
            # cv2.waitKey(0)

    return fusion_image


# ------------ YOLOv5 -----------------
import torch
import torchvision

from utils.general import scale_boxes, xywh2xyxy  # , non_max_suppression
from utils.segment.general import process_mask, scale_image
from models.yolo import DetectMultiBackend


def initYOLOModel(task: str = 'object'):
    """
    初始化YOLO模型，返回YOLOv5模型

    输入：
        task：选择**目标检测模型**或者**实例分割模型**，可以接受`object`或`segment`参数
    输出：
        返回对应的YOLOv5模型
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if task == 'object':
        # YOLOv5模型
        model = DetectMultiBackend(weights='./weights/yolov5s.pt',
                                   data='./data/coco.yaml', device=device)
    elif task == 'segment':
        model = DetectMultiBackend(
            weights='./weights/yolov5x-seg.pt', device=device,
            data='./data/coco128.yaml')
    else:
        Exception('task not exit ', task)
    return model


def runModel(model, image, detect_type='object'):
    """
    运行模型，返回执行结果

    输入：
        model：YOLO模型，可以通过 `initYOLOModel` 获取
        image：图片数组，可使用cv2或者plt获取对应的数组

    输出：
        返回对应图片检测的结果
    """
    # 图片处理
    img = letterbox(image)[0]  # padded resize
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)  # contiguous
    img = torch.from_numpy(img).to(model.device)
    img = img.half() if model.fp16 else img.float()  # uint8 to fp16/32
    img /= 255  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim

    # 执行检测
    if detect_type == 'object':
        pred = model(img)
        pred = non_max_suppression(pred)
    elif detect_type == 'segment':
        pred, proto = model(img)[:2]
        pred = non_max_suppression(pred, nm=32)
    else:
        Exception('type is error, only supposed `object` or `segment`')

    result = []
    masks = None
    # 获取检测/分割结果
    for i, det in enumerate(pred):  # batch size
        # 获取掩码
        if detect_type == 'segment':
            masks = process_mask(proto[i], det[:, 6:], det[:, :4], img.shape[2:], upsample=True)  # HWC
            masks = masks.permute(1, 2, 0).contiguous()
            masks = masks.cpu().numpy()
            masks = scale_image(masks.shape[:2], masks, image.shape)
            # 将masks进行置信度过滤
            mask_conf = 0.45
            masks[masks >= mask_conf] = 1
            masks[masks <= mask_conf] = 0
            masks = masks.astype(np.uint8)

        # 将获取的包围盒缩放至图片大小
        det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], image.copy().shape).round()
        # 获取每个对象的坐标等结果
        for j, (*xyxy, conf, cls) in enumerate(det[:, :6]):
            xyxy = [pos.cpu().numpy() for pos in xyxy]
            conf = conf.cpu().numpy()
            if masks is not None:
                result.append([xyxy, conf, cls, cv2whc(masks[:, :, j:j + 1])])
            else:
                result.append([xyxy, conf, cls, None])

    return result, det


def show_detection_result(image, detect_type: str = 'object'):
    model = initYOLOModel(detect_type)
    result, prediction_matrix = runModel(model, cv2whc(image), detect_type)

    # import cv2

    # 获取所有标签
    confs = prediction_matrix[:, 4:5]

    # names 标签中可能包含 ['0: person', '1: person', '2: tie', '3: tie']
    names = [str(i) + ': ' + model.names[val] for i, val in enumerate(prediction_matrix[:, 5:6])]

    # 对包装好的单个内容进行显示
    for i, obj in enumerate(result):
        # 解包内容
        xyxy, conf, cls, mask = obj
        name = str(i) + ': ' + model.names[int(cls)]

        if name not in names:
            continue

        # cv2.imshow(name, cv2whc(image[xyxy[0]:xyxy[2], xyxy[1]:xyxy[3]]))
        # cv2.waitKey(0)
        pass


def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        max_det=300,
        nm=0,  # number of masks
):
    """
    非极大值抑制，合并IoU较大的预测框，最大化加密
    prediction的格式为 [batch_size, grid_cell, location+confidence+classes] e.g. 1,15120,
    1. 过滤 存在物体的置信度 小于conf_thres的框
    2. 过滤 目标置信度 小于conf_thres的框
    """

    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes 类别个数
    xc = prediction[..., 4] > conf_thres  # candidates 获取置信度大于conf_thres的检测框

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    mi = 5 + nc  # mask start index # mask的开始索引
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[
            xc[xi]]  # confidence 置信度数组 shape = (64, 117) 表示只有64个置信度大于conf_thres的物体，其中的117表示x,y,h,w,conf,80 cls,32 mask

        # If none remain process next image
        if not x.shape[0]:  # 若全部置信度都比较低，可能是当前图片中不存在物体
            continue

        # Compute conf 目标置信度
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks # 获取mask

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)  # 获取目标置信度最大的 conf和对应索引 shape = (64,1)
            x = torch.cat((box, conf, j.float(), mask), 1)[
                conf.view(-1) > conf_thres]  # 将80个类别化成了1个数字（类别对应的索引） shape=(60,38)，此时的xywh转为了xyxy

        # Filter by class （过滤掉不需要的类别，需要的类别存放在classes中，以数字形式）
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes （最后得到的可能包含物体的包围盒的数量）
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
        else:
            x = x[x[:, 4].argsort(descending=True)]  # sort by confidence 按置信度从高到低排序

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        # boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores（不知道为什么要加上类别偏移）
        # i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS 非极大值抑制（返回最优boxes的索引）
        boxes, scores = x[:, :4], x[:, 4]

        boxes_num = boxes.shape[0]  # 获取包围盒的数量

        is_same_box = np.ones((boxes_num,), dtype=np.int32) * -1  # 若不为-1则表示该物体和其数据对应的物体是同一个

        # 获取最大的boxes
        for i in range(0, boxes_num):
            if is_same_box[i] != -1: continue  # 排除掉相同物体遍历
            # is_same_box[i] = i # 做unique耗时，改为这个
            lt = boxes[i][:2]  # 第i个盒子的左上角坐标
            rb = boxes[i][2:]  # 第i个盒子的右下角坐标
            for j in range(i + 1, boxes_num):
                if is_same_box[j] != -1: continue
                bx1 = boxes[i].reshape((1, -1))
                bx2 = boxes[j].reshape((1, -1))
                iou = torchvision.ops.boxes.box_iou(bx1, bx2)
                if iou > iou_thres:  # 大于阈值，表示两个物体相交，且相交的范围较大（即可能为同一物体的可能性很大）
                    # print(i, ' is same of ', j, ' with iou ', iou)
                    lt = torch.min(lt, boxes[j][:2])
                    rb = torch.max(rb, boxes[j][2:])
                    is_same_box[j] = i
            x[i, :4] = torch.tensor(([lt[0], lt[1], rb[0], rb[1]]), dtype=x.dtype)

        # 设置返回值
        output[xi] = x[is_same_box == -1, :]

    return output
