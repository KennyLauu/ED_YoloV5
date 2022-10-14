import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
# matplotlib==3.5.2
from Encryption.noColorDecry import noColorDecry
from Encryption.noColorEncry import noColorEncry
from Encryption.EncryUtils import ProcessingKey

st.set_option('deprecation.showfileUploaderEncoding', False)

# Upload an image and set some options for demo purposes
st.header("自定义加密")
img_file = st.sidebar.file_uploader(label='上传一张图片', type=['png', 'jpg'])
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
    # Get a cropped image from the frontend
    cropped_img = st_cropper(img, realtime_update=realtime_update, box_color=box_color,
                             aspect_ratio=aspect_ratio, return_type='box')
    xyxy = [cropped_img['left'], cropped_img['top'], cropped_img['left'] + cropped_img['width'], cropped_img['top'] + cropped_img['height']]
    print(xyxy)
    st.write(xyxy)
    # Manipulate cropped image at will
    # st.write("预览截图")
    # _ = cropped_img.thumbnail((150, 150))
    st.image(np.array(img)[xyxy[0]:xyxy[2], xyxy[1]:xyxy[3]])
