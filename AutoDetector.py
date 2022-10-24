import numpy as np
import torch

# from DetectUtils import non_max_suppression
from models.experimental import attempt_load
from tracker import update_tracker
from utils.augmentations import letterbox
from utils.general import scale_boxes, non_max_suppression
from utils.torch_utils import select_device
from utils.segment.general import process_mask, scale_image


class baseDet(object):

    def __init__(self):
        self.img_size = 640
        self.threshold = 0.3
        self.stride = 1

    def build_config(self):
        self.faceTracker = {}
        self.faceClasses = {}
        self.faceLocation1 = {}
        self.faceLocation2 = {}
        self.frameCounter = 0
        self.currentCarID = 0
        self.recorded = []

        # self.font = cv2.FONT_HERSHEY_SIMPLEX

    def feedCap(self, im):
        retDict = {
            'frame': None,
            'faces': None,
            'list_of_ids': None,
            'face_bboxes': []
        }
        self.frameCounter += 1

        # im, faces, face_bboxes = update_tracker(self, im, self.frameCounter)
        im, trackers = update_tracker(self, im, self.frameCounter)

        retDict['frame'] = im
        retDict['tracker'] = trackers
        # retDict['faces'] = faces
        # retDict['face_bboxes'] = face_bboxes

        return retDict

    def init_model(self):
        raise EOFError("Undefined model type.")

    def preprocess(self, _):
        raise EOFError("Undefined model type.")

    def detect(self, _):
        raise EOFError("Undefined model type.")


class Detector(baseDet):

    def __init__(self):
        super(Detector, self).__init__()
        # self.init_model()
        self.build_config()

    def init_model(self, weight: str = 'weights/yolov5x.pt', detect_type='object'):
        self.detect_type = detect_type
        self.weights = weight
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        self.device = select_device(self.device)
        model = attempt_load(self.weights, device=self.device)
        model.to(self.device).eval()
        model.half()  # if model.fp16 else model.float()
        # torch.save(model, 'test.pt')
        self.m = model
        self.names = model.module.names if hasattr(
            model, 'module') else model.names

    def preprocess(self, img):

        img0 = img.copy()
        img = letterbox(img, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half()  # if self.m.fp16 else img.float()  # uint8 to fp16/32
        img /= 255.0  # 图像归一化
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img0, img

    def detect(self, im):

        im0, img = self.preprocess(im)

        if self.detect_type == 'object':
            pred = self.m(img, augment=False)[0]
            pred = pred.float()
            pred = non_max_suppression(pred, self.threshold, 0.4)
        else:
            pred, proto = self.m(img)[:2]
            pred = non_max_suppression(pred, nm=32)

        pred_boxes = []
        masks = None
        for det in pred:

            if det is not None and len(det):
                if self.detect_type == 'segment':
                    masks = process_mask(proto[0], det[:, 6:], det[:, :4], img.shape[2:], upsample=True)  # HWC
                    masks = masks.permute(1, 2, 0).contiguous()
                    masks = masks.cpu().numpy()
                    masks = scale_image(masks.shape[:2], masks, im0.shape)
                    # 将masks进行置信度过滤
                    mask_conf = 0.1
                    masks[masks >= mask_conf] = 1
                    masks[masks <= mask_conf] = 0
                    masks = masks.astype(np.uint8)

                det[:, :4] = scale_boxes(
                    img.shape[2:], det[:, :4], im0.shape).round()

                for *x, conf, cls_id in det[:, :6]:
                    lbl = self.names[int(cls_id)]
                    # if not lbl in ['person', 'car', 'truck']:
                    #     continue
                    x1, y1 = int(x[0]), int(x[1])
                    x2, y2 = int(x[2]), int(x[3])
                    pred_boxes.append(
                        (x1, y1, x2, y2, lbl, conf))

        return im, pred_boxes, masks
