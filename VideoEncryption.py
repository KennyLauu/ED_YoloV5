import cv2
import imutils
from AutoDetector import Detector
import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def get_frames(frame_path, track_detect):
    track_detect.is_label = True
    track_detect.update_tracker()
    cap = cv2.VideoCapture(frame_path)
    all_frames = cap.get(7)
    if all_frames < 4:
        return None, None
    frame = 1
    image = None
    track = None


    while True:
        # try:
        ret, im = cap.read()
        print(frame)

        if ret is False or im is None:
            break
        if frame > 1:
            break

        result = track_detect.feedCap(im)
        image = result['frame']
        track = result['tracker']
        frame += 1

    return image, track

def encryption_video_with_(track_detect, label_id, frame_path, frame_status):
    """
    track_detect 为检测器
    label_id 为需要加密的id
    frame path 为视频路径
    """
    print('encryption video with ', frame_path)
    # 设置加密的标签
    track_detect.set_encryption_obj(label_id)

    cap = cv2.VideoCapture(frame_path)
    fps = int(cap.get(5))  # 获取视频帧率
    frame_status[1] = cap.get(7)
    videoWriter = None
    try:
        while True:
            ret, im = cap.read()
            print('processing: ', frame_status[0], '/', frame_status[1])

            if ret is False or im is None:
                break

            result = track_detect.feedCap(im)
            image = result['frame']
            frame_status[0] += 1

            if videoWriter is None:
                fourcc = cv2.VideoWriter_fourcc(
                    'm', 'p', '4', 'v')  # opencv3.0
                videoWriter = cv2.VideoWriter(
                    'encryption_output.mp4', fourcc, fps, (image.shape[1], image.shape[0]))

            videoWriter.write(image)
            # cv2.imshow('name', image)
            # cv2.waitKey(int(1000 / fps))
            #
            # if cv2.getWindowProperty('name', cv2.WND_PROP_AUTOSIZE) < 1:
            #     # 点x退出
            #     break

    finally:
        cap.release()
        videoWriter.release()
        cv2.destroyAllWindows()

