import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image
from Encryption.EncryUtils import *
from DetectUtils import *
from utils.plots import Annotator
from utils.plots import colors

st.set_option('deprecation.showfileUploaderEncoding', False)

model_detect, model_segment = None, None
if model_detect is None:
    model_detect = initYOLOModel('object')
if model_segment is None:
    model_segment = initYOLOModel('segment')


def app():
    option = st.sidebar.selectbox(
        '请选择加密的模式',
        ('全图加密', '选择性加密', '基于目标检测加密', '基于实例分割加密'))

    if option == '全图加密':
        st.header('全图加密模式')

        img_file = st.sidebar.file_uploader(label='上传一张需要加密图片', type=['png', 'jpg'])
        if img_file:
            with st.container():
                contact_form_left, contact_form_right = st.columns((1, 1), gap='medium')
                img = Image.open(img_file)
                with contact_form_left:
                    st.subheader('原图片')
                    img.save('../data/images/pendingImg.png')
                    st.image(img)
                with contact_form_right:
                    st.subheader('加密后的图片')
                    img = np.ascontiguousarray(img)
                    key = ProcessingKey(img)
                    EncryImg = noColorEncry(img, key)

                    encryImg = Image.fromarray(EncryImg)
                    encryImg.save('../data/images/self_encryImg.png')
                    st.image(encryImg)
                    # DecryImg = noColorDecry(EncryImg, key)
                    with open('../data/images/self_encryImg.png', 'rb') as file:
                        btn = st.sidebar.download_button(
                            label='下载加密后的图片',
                            data=file,
                            file_name='Encrytion_image.png',
                            mime="image/png"
                        )
                    st.sidebar.subheader('图片提取码')
                    st.sidebar.write(str(key[0]), str(key[1]), str(key[2]), str(key[3]))

    elif option == '选择性加密':
        st.header("自定义加密")
        img_file = st.sidebar.file_uploader(label='上传一张需要加密图片', type=['png', 'jpg'])
        realtime_update = st.sidebar.checkbox(label="实时更新", value=True)
        box_color = st.sidebar.color_picker(label="锚框颜色", value='#0000FF')
        aspect_choice = st.sidebar.radio(label="长宽比", options=["1:1", "16:9", "4:3", "2:3", "Free"])
        aspect_dict = {
            "1:1": (1, 1),
            "16:9": (16, 9),
            "4:3": (4, 3),
            "2:3": (2, 3),
            "Free": None
        }
        aspect_ratio = aspect_dict[aspect_choice]

        if img_file:
            img = Image.open(img_file)
            if not realtime_update:
                st.write("点击两下保存")
            with st.container():
                contact_form_left, contact_form_right = st.columns((2, 1), gap='small')

                with contact_form_left:
                    st.subheader('原图像')
                    cropped_img = st_cropper(img, realtime_update=realtime_update, box_color=box_color,
                                             aspect_ratio=aspect_ratio, return_type='box')
                    xyxy = [cropped_img['left'], cropped_img['top'], cropped_img['left'] + cropped_img['width'],
                            cropped_img['top'] + cropped_img['height']]
                    img = np.ascontiguousarray(img)
                    key = KeyGenerator(img)
                    _, fusion_img = SelectAreaEcryption(cv2whc(img), xyxy, key)
                    Image.fromarray(cv2whc(fusion_img)).save('../data/images/custom_encryImg.png')
                    with open('../data/images/custom_encryImg.png', 'rb') as file:
                        btn = st.download_button(
                            label='下载加密后的图片',
                            data=file,
                            file_name='Encrytion_image.png',
                            mime="image/png"
                        )
                    st.subheader('图片提取码')
                    st.write(str(key[0]), str(key[1]), str(key[2]), str(key[3]))

                with contact_form_right:
                    st.subheader('自定义图像预览')
                    st.image(np.array(img)[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]])
                    # Manipulate cropped image at will
                    # st.write("预览截图")
                    # _ = cropped_img.thumbnail((150, 150))
                    st.write('自定义图像尺寸大小')
                    st.write(xyxy[2] - xyxy[0], ' x ', xyxy[3] - xyxy[1])
                    st.image(cv2whc(fusion_img))

    elif option == '基于目标检测加密':
        st.header('基于目标检测的加密模式')
        img_file = st.sidebar.file_uploader(label='上传一张需要加密图片', type=['png', 'jpg'])
        if img_file:
            with st.container():
                contact_form_left, contact_form_right = st.columns((2, 2), gap='medium')
                img = Image.open(img_file)
                with contact_form_left:
                    st.subheader('原图片')
                    img.save('../data/images/pendingImg.png')

                    img = np.ascontiguousarray(img)
                    annotator = Annotator(img.copy(), line_width=2)
                    key = ProcessingKey(img)
                    result, notuse_value = runModel(model_detect, img)
                    # fusion_image = img

                    multiselect = st.sidebar.multiselect('你需要加密的类别',
                                                         [str(i) + ': ' + model_detect.names[int(v)] for i, v in
                                                          enumerate(notuse_value[:, 5:6])])

                    fusion_image = cv2whc(img)

                    # ------------
                    # 全局操作
                    stack.clear()
                    # ------------

                    # 对于每个检测出来的物体
                    for i, obj in enumerate(result):
                        # 解包内容
                        xyxy, conf, cls, mask = obj
                        name = str(i) + ': ' + model_detect.names[int(cls)]
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
                    im0 = annotator.result()
                    st.image(im0, use_column_width='always')

                with contact_form_right:
                    st.subheader('加密后的图片')

                    st.image(cv2whc(fusion_image), use_column_width='always')
                    Image.fromarray(cv2whc(fusion_image)).save('../data/images/detect_encryImg.png')
                    with open('../data/images/detect_encryImg.png', 'rb') as file:
                        btn = st.sidebar.download_button(
                            label='下载加密后的图片',
                            data=file,
                            file_name='Encrytion_image.png',
                            mime="image/png"
                        )
                    st.sidebar.subheader('图片提取码')
                    st.sidebar.write(str(key[0]), str(key[1]), str(key[2]), str(key[3]))

    elif option == '基于实例分割加密':
        st.header('基于实例分割的加密模式')
        img_file = st.sidebar.file_uploader(label='上传一张需要加密图片', type=['png', 'jpg'])
        if img_file:
            with st.container():
                contact_form_left, contact_form_right = st.columns((2, 2), gap='medium')
                img = Image.open(img_file)
                with contact_form_left:
                    st.subheader('原图片')
                    img.save('../data/images/pendingImg.png')

                    img = np.ascontiguousarray(img)
                    annotator = Annotator(img.copy(), line_width=2)
                    key = ProcessingKey(img)
                    result, notuse_value = runModel(model_segment, img, 'segment')
                    # fusion_image = img

                    multiselect = st.sidebar.multiselect('你需要加密的类别',
                                                         [str(i) + ': ' + model_segment.names[int(v)] for i, v in
                                                          enumerate(notuse_value[:, 5:6])])
                    fusion_image = cv2whc(img)
                    # if multiselect:
                    #     print('0', multiselect)

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
                    im0 = annotator.result()
                    st.image(im0, use_column_width='always')

            with contact_form_right:
                st.subheader('加密后的图片')

                st.image(cv2whc(fusion_image), use_column_width='always')
                Image.fromarray(cv2whc(fusion_image)).save('../data/images/segment_encryImg.png')
                with open('../data/images/segment_encryImg.png', 'rb') as file:
                    btn = st.sidebar.download_button(
                        label='下载加密后的图片',
                        data=file,
                        file_name='Encrytion_image.png',
                        mime="image/png"
                    )
            st.sidebar.subheader('图片提取码')
            st.sidebar.write(str(key[0]), str(key[1]), str(key[2]), str(key[3]))
