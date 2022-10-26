import os
import sys
import cv2
import imageio.v3 as iio
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
import time
from AutoDetector import Detector
import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image
from Encryption.noColorEncry import noColorEncry
from Encryption.EncryUtils import *
from DetectUtils import *
from utils.plots import Annotator
from utils.plots import colors
from VideoEncryption import *
from copy import deepcopy

st.set_option('deprecation.showfileUploaderEncoding', False)

model_detect, model_segment, track_detect = None, None, None
if model_detect is None:
    model_detect = initYOLOModel('object')
if model_segment is None:
    model_segment = initYOLOModel('segment')
if track_detect is None:
    track_detect = Detector()
    track_detect.init_model()


def numpy_array_to_video(numpy_array, video_out_path):
    video_height = numpy_array.shape[1]
    video_width = numpy_array.shape[2]

    out_video_size = (video_width, video_height)
    output_video_fourcc = int(cv2.VideoWriter_fourcc(*'mp4v'))
    video_write_capture = cv2.VideoWriter(video_out_path, output_video_fourcc, 30, out_video_size)

    for frame in numpy_array:
        video_write_capture.write(frame)

    video_write_capture.release()

repeat_process = False
frame_img = None
tracks_process = None
name_id_dict = None

def app():

    option = st.sidebar.selectbox(
        '请选择加密的模式',
        # ('全图加密', '选择性加密', '基于目标检测加密', '基于实例分割加密'))
        ('自定义加密', '基于实例分割加密', '视频分割'))

    if option == '全图加密':
        pass
    #     st.header('全图加密模式')
    #
    #     img_file = st.sidebar.file_uploader(label='上传一张需要加密图片', type=['png', 'jpg'])
    #     if img_file:
    #         with st.container():
    #             contact_form_left, contact_form_right = st.columns((1, 1), gap='medium')
    #             img = Image.open(img_file)
    #             with contact_form_left:
    #                 st.subheader('原图片')
    #                 img.save('../data/images/pendingImg.png')
    #                 st.image(img)
    #             with contact_form_right:
    #                 st.subheader('加密后的图片')
    #                 img = np.ascontiguousarray(img)
    #                 key = ProcessingKey(img)
    #                 EncryImg = noColorEncry(img, key)
    #
    #                 # encryImg = Image.fromarray(np.transpose(EncryImg, (1, 0, 2)))
    #                 encryImg = Image.fromarray(EncryImg)
    #                 encryImg.save('../data/images/self_encryImg.png')
    #                 st.image(encryImg)
    #                 # DecryImg = noColorDecry(EncryImg, key)
    #                 with open('../data/images/self_encryImg.png', 'rb') as file:
    #                     btn = st.sidebar.download_button(
    #                         label='下载加密后的图片',
    #                         data=file,
    #                         file_name='Encrytion_image.png',
    #                         mime="image/png"
    #                     )
    #                 st.sidebar.subheader('图片提取码')
    #                 st.sidebar.write(str(key[0]), str(key[1]), str(key[2]), str(key[3]))

    elif option == '自定义加密':
        st.header("自定义加密")
        img_file = st.file_uploader(label='上传一张需要加密图像', type=['png', 'jpg'])
        placeholder = st.empty()
        with placeholder.container():
            with st.container():
                init_form_left, init_form_right = st.columns((1, 1), gap='large')
                with init_form_left:
                    st.subheader('原图像')
                    st.image('../data/images/plainimg.png', width=500)
                with init_form_right:
                    st.subheader('加密后的图像')
                    st.image('../data/images/cipherimg.png', width=500)

        if img_file:
            placeholder.empty()
            img = Image.open(img_file)
            # realtime_update = st.sidebar.checkbox(label="实时更新", value=True)
            style_form_left, style_form_right = st.columns((1, 1), gap='medium')
            with style_form_left:
                aspect_choice = st.radio(label="长宽比", options=["1:1", "16:9", "4:3", "2:3", "Free", "全图"],
                                         horizontal=True)
            with style_form_right:
                box_color = st.color_picker(label="锚框颜色", value='#0000FF')
            aspect_dict = {
                "1:1": (1, 1),
                "16:9": (16, 9),
                "4:3": (4, 3),
                "2:3": (2, 3),
                "Free": None,
                "全图": None
            }
            aspect_ratio = aspect_dict[aspect_choice]
            # if not realtime_update:
            #     st.write("点击两下保存")
            with st.container():
                contact_form_left, contact_form_right = st.columns((2, 1), gap='small')

                with contact_form_left:
                    st.subheader('原图像')
                    if aspect_choice == '全图':
                        xyxy = [0, 0, img.width, img.height]
                        st.image(img)
                    else:
                        cropped_img = st_cropper(img, box_color=box_color,
                                                 aspect_ratio=aspect_ratio, return_type='box')
                        xyxy = [cropped_img['left'], cropped_img['top'], cropped_img['left'] + cropped_img['width'],
                                cropped_img['top'] + cropped_img['height']]
                    img = np.ascontiguousarray(img)
                    key = ProcessingKey(img)
                    img = PIL2whc(img)
                    print(key)
                    encryption_object, fusion_img = SelectAreaEncryption(img, xyxy, key)
                    Image.fromarray(PIL2whc(fusion_img)).save('../data/images/custom_encryImg.png')
                    # 写入加密需要的信息
                    SetEncryptionImage('../data/images/custom_encryImg.png', encryption_object, 'custom', fusion_img,
                                       key)
                    with open('../data/images/custom_encryImg.png', 'rb') as file:
                        btn = st.download_button(
                            label='下载加密后的图像',
                            data=file,
                            file_name='Encrytion_image.png',
                            mime="image/png"
                        )

                    st.subheader('图像提取码')
                    st.write(str(key[0]), str(key[1]), str(key[2]), str(key[3]))

                with contact_form_right:
                    st.subheader('自定义图像预览')
                    st.image(PIL2whc(img)[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]])
                    # Manipulate cropped image at will
                    # st.write("预览截图")
                    # _ = cropped_img.thumbnail((150, 150))
                    st.write('自定义图像尺寸大小')
                    st.write(xyxy[2] - xyxy[0], ' x ', xyxy[3] - xyxy[1])
                    st.image(PIL2whc(fusion_img))

    # elif option == '基于目标检测加密':
    #     st.header('基于目标检测的加密模式')
    #     img_file = st.sidebar.file_uploader(label='上传一张需要加密图片', type=['png', 'jpg'])
    #     if img_file:
    #         with st.container():
    #             contact_form_left, contact_form_right = st.columns((2, 2), gap='medium')
    #             img = Image.open(img_file)
    #             with contact_form_left:
    #                 st.subheader('原图片')
    #                 img.save('../data/images/pendingImg.png')
    #
    #                 img = np.ascontiguousarray(img)
    #                 annotator = Annotator(img.copy(), line_width=2)
    #                 key = ProcessingKey(img)
    #                 result, notuse_value = runModel(model_detect, img)
    #                 # fusion_image = img
    #
    #                 multiselect = st.sidebar.multiselect('你需要加密的类别',
    #                                                      [str(i) + ': ' + model_detect.names[int(v)] for i, v in
    #                                                       enumerate(notuse_value[:, 5:6])])
    #
    #                 encryption_object = []
    #                 fusion_image = cv2whc(img)
    #
    #                 if len(result) == 0:
    #                     fusion_image = noColorEncry(fusion_image, key)
    #
    #                     # encryImg = Image.fromarray(np.transpose(EncryImg, (1, 0, 2)))
    #                     encryImg = Image.fromarray(cv2whc(fusion_image))
    #                     encryImg.save('../data/images/detect_encryImg.png')
    #
    #                 # ------------
    #                 # 全局操作
    #                 stack.clear()
    #                 # ------------
    #
    #                 # 对于每个检测出来的物体
    #                 for i, obj in enumerate(result):
    #                     # 解包内容
    #                     xyxy, conf, cls, mask = obj
    #                     name = str(i) + ': ' + model_detect.names[int(cls)]
    #                     annotator.box_label(xyxy, name, color=colors(int(cls), True))
    #                     if name not in multiselect:
    #                         continue
    #                     # 重叠判定（若之前存在已加密的内容，则当前物体存在部分不需要加密）
    #                     is_overlap, overlap_areas = Overlap(xyxy, mask)
    #                     encryption_image, mask, fusion_image = OverlapEncryption(fusion_image, xyxy, key,
    #                                                                              overlap_areas,
    #                                                                              mask, name) \
    #                         if is_overlap else \
    #                         DirectEncryption(fusion_image, xyxy, key, mask, name)
    #                     encryption_object.append([encryption_image, xyxy, mask])
    #                 im0 = annotator.result()
    #                 st.image(im0, use_column_width='always')
    #
    #             with contact_form_right:
    #                 st.subheader('加密后的图片')
    #
    #                 st.image(cv2whc(fusion_image), use_column_width='always')
    #                 Image.fromarray(cv2whc(fusion_image)).save('../data/images/detect_encryImg.png')
    #                 # 写入加密需要的信息
    #                 SetEncryptionImage('../data/images/detect_encryImg.png', encryption_object, 'object', fusion_image)
    #                 with open('../data/images/detect_encryImg.png', 'rb') as file:
    #                     btn = st.sidebar.download_button(
    #                         label='下载加密后的图片',
    #                         data=file,
    #                         file_name='Encrytion_image.png',
    #                         mime="image/png"
    #                     )
    #                 st.sidebar.subheader('图片提取码')
    #                 st.sidebar.write(str(key[0]), str(key[1]), str(key[2]), str(key[3]))

    elif option == '基于实例分割加密':
        st.header('基于实例分割的加密模式')
        img_file = st.file_uploader(label='上传一张需要加密图像', type=['png', 'jpg'])
        placeholder = st.empty()
        with placeholder.container():
            with st.container():
                init_form_left, init_form_right = st.columns((1, 1), gap='large')
                with init_form_left:
                    st.subheader('原图像')
                    st.image('../data/images/plainimg.png', width=500)
                with init_form_right:
                    st.subheader('加密后的图片')
                    st.image('../data/images/cipherimg.png', width=500)

        if img_file:
            placeholder.empty()
            with st.container():
                contact_form_left, contact_form_right = st.columns((2, 2), gap='medium')
                img = Image.open(img_file)
                with contact_form_left:
                    st.subheader('原图片')
                    img.save('../data/images/pendingImg.png')

                    img = np.ascontiguousarray(img)
                    annotator = Annotator(img.copy(), line_width=2)
                    key = ProcessingKey(img)
                    print(key)
                    result, notuse_value = runModel(model_segment, img, 'segment')
                    # fusion_image = img

                    multiselect = st.sidebar.multiselect('你需要加密的类别',
                                                         [str(i) + ': ' + model_segment.names[int(v)] for i, v in
                                                          enumerate(notuse_value[:, 5:6])])
                    encryption_object = []
                    fusion_image = PIL2whc(img)
                    # if multiselect:
                    #     print('0', multiselect)
                    if len(result) == 0:
                        fusion_image = noColorEncry(fusion_image, key)

                        # encryImg = Image.fromarray(np.transpose(EncryImg, (1, 0, 2)))
                        encryImg = Image.fromarray(fusion_image)
                        encryImg.save('../data/images/segment_encryImg.png')

                    # ------------
                    # 全局操作
                    stack.clear()
                    # ------------

                    # 对于每个检测出来的物体
                    for i, obj in enumerate(result):
                        # 解包内容
                        xyxy, conf, cls, mask = obj
                        name = str(i) + ': ' + model_segment.names[int(cls)]
                        annotator.box_label(xyxy, name, color=colors(int(cls), True))

                        if name not in multiselect:
                            continue
                        # 重叠判定（若之前存在已加密的内容，则当前物体存在部分不需要加密）
                        is_overlap, overlap_areas = Overlap(xyxy, mask)
                        encryption_image, mask, fusion_image = OverlapEncryption(fusion_image, xyxy, key,
                                                                                 overlap_areas,
                                                                                 mask, name) \
                            if is_overlap else \
                            DirectEncryption(fusion_image, xyxy, key, mask, name)
                        encryption_object.append([encryption_image, xyxy, mask])
                    im0 = annotator.result()
                    st.image(im0, use_column_width='always', width=850)

            with contact_form_right:
                st.subheader('加密后的图像')

                st.image(PIL2whc(fusion_image), use_column_width='always')
                Image.fromarray(PIL2whc(fusion_image)).save('../data/images/segment_encryImg.png')
                # 写入加密需要的信息
                SetEncryptionImage('../data/images/segment_encryImg.png', encryption_object, 'segment', fusion_image,
                                   key)
                with open('../data/images/segment_encryImg.png', 'rb') as file:
                    btn = st.sidebar.download_button(
                        label='下载加密后的图像',
                        data=file,
                        file_name='Encrytion_image.png',
                        mime="image/png"
                    )
            st.sidebar.subheader('图像提取码')
            st.sidebar.write(str(key[0]), str(key[1]), str(key[2]), str(key[3]))

    elif option == '视频分割':
        st.subheader('视频分割模块')
        # 此处将字节流处理成视频格式 具体参考
        # https://stackoverflow.com/questions/60558412/how-to-decode-a-video-memory-file-byte-string-and-step-through-it-frame-by-f
        # pip install imageio[ffmpeg]
        uploaded_file = st.file_uploader("Choose a file", type=["mp4", "avi"], accept_multiple_files=False)
        placeholder = st.empty()
        with placeholder.container():
            with st.container():
                init_form_left, init_form_right = st.columns((1, 1), gap='large')
                with init_form_left:
                    st.subheader('原视频')
                    st.image('../data/images/plainvideo.png', width=500)
                with init_form_right:
                    st.subheader('加密后的视频')
                    st.image('../data/images/ciphervideo.png', width=500)
        if uploaded_file:
            placeholder.empty()
            with st.container():
                contact_form_left, contact_form_right = st.columns((2, 2), gap='medium')
                bytes_data = uploaded_file.read()
                frames = iio.imread(bytes_data, index=None)
                frames = np.array(frames)

                global repeat_process, frame_img, tracks_process, name_id_dict
                # tracks_process.tracks.clear()
                numpy_array_to_video(np.ascontiguousarray(frames[:, :, :, ::-1]), '../data/videos/outputvideo.mp4')
                frame_img, tracks = get_frames('../data/videos/outputvideo.mp4', track_detect)
                tracks_process = deepcopy(tracks)
                name_id_dict = {'{}: {}'.format(t.track_id, t.cls_): t.track_id for t in tracks_process.tracks}
                with contact_form_left:
                    st.subheader('原视频')
                    muti_cls = st.multiselect(
                        '你需要加密的类别',
                        ['{}: {}'.format(t.track_id, t.cls_) for t in tracks_process.tracks]
                    )
                    st.image(frame_img[:, :, ::-1])
                #  获取选择的物体
                with contact_form_right:
                    st.subheader('加密后的视频')
                    st.write("##")
                    select_id = []
                    for val in muti_cls:
                        select_id.append(name_id_dict[val])

                    frame_status = [0, 1]
                    if len(muti_cls) != 0:
                        # 设置加密的标签
                        track_detect.set_encryption_obj(select_id)
                        cap = cv2.VideoCapture('../data/videos/outputvideo.mp4')
                        fps = int(cap.get(5))  # 获取视频帧率
                        frame_status[1] = cap.get(7)
                        videoWriter = None
                        placeholder = st.empty()
                        with st.spinner('正在处理中，请稍等...'):
                            my_bar = placeholder.progress(0)
                            try:
                                while True:
                                    ret, im = cap.read()
                                    print('processing: ', frame_status[0], '/', frame_status[1])
                                    my_bar.progress(frame_status[0] / frame_status[1])
                                    if ret is False or im is None:
                                        break

                                    result = track_detect.feedCap(im)
                                    image = result['frame']
                                    frame_status[0] += 1

                                    if videoWriter is None:
                                        fourcc = cv2.VideoWriter_fourcc(
                                            'm', 'p', '4', 'v')  # opencv3.0
                                        videoWriter = cv2.VideoWriter(
                                            '../data/videos/encryption_output.mp4', fourcc, fps, (image.shape[1], image.shape[0]))

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
                        st.success('已完成，请点击按钮下载!')
                        video_file = open('../data/videos/encryption_output.mp4', 'rb')
                        video_bytes = video_file.read()
                        # video_np = iio.imread(video_bytes, index=None)
                        # video_np = np.array(video_np)
                        # print(video_np)
                        st.balloons()
                        st.video(video_bytes)

                        with open('../data/videos/encryption_output.mp4', 'rb') as file:
                            btn = st.download_button(
                                label='下载加密后的图像',
                                data=file,
                                file_name='Encrytion_video.mp4',
                                mime="video/mp4"
                            )
                        placeholder.empty()







