import time

import cv2
import imutils

from AutoDetector import Detector


def main():
    try:
        name = 'demo'

        det = Detector()
        det.init_model(weight='weights/yolov5s-seg.pt', detect_type='segment')
        # det.init_model(weight='weights/yolov5s.pt', detect_type='object')

        cap = cv2.VideoCapture('D:/User/Documents/Tencent Files/2696761655/FileRecv/demo01.mp4')
        # cap.set(cv2.CAP_PROP_POS_FRAMES, 1480)  # 设置要获取的帧号
        fps = int(cap.get(5))  # 获取视频帧率
        all_frames = cap.get(7)  # 获取所有帧数量
        # print('fps:', fps)
        t = int(1000 / fps)
        start = time.time()

        videoWriter = None
        frame = 1

        while True:
            inner_start = time.time()
            # try:
            ret, im = cap.read()
            if ret is False or im is None:
                break

            result = det.feedCap(im)
            result = result['frame']
            result = imutils.resize(result, height=500)
            if videoWriter is None:
                fourcc = cv2.VideoWriter_fourcc(
                    'm', 'p', '4', 'v')  # opencv3.0
                videoWriter = cv2.VideoWriter(
                    'result.mp4', fourcc, fps, (result.shape[1], result.shape[0]))

            end = time.time()
            print('total time: ', end - inner_start)
            print(frame, '/', all_frames, ' fps: ', frame / (end - start))
            frame += 1

            videoWriter.write(result)
            cv2.imshow(name, result)
            cv2.waitKey(t)

            if cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) < 1:
                # 点x退出
                break
            # except Exception as e:
            #     print(e)
            #     break
    finally:
        print('total time: ', time.time() - start)
        cap.release()
        videoWriter.release()
        cv2.destroyAllWindows()
    print('total time: ', time.time() - start)
    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
