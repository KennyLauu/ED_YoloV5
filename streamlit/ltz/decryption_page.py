import os
import sys

import numpy as np
import streamlit as st
from PIL import Image
import base64

from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from Decryption_img import Decryption_img
from DetectUtils import *


def app():
    st.header('欢迎来到解密模块')
    img_file = st.file_uploader(label='上传一张需要解密的图片', type=['png', 'jpg'])
    placeholder = st.empty()
    with placeholder.container():
        with st.container():
            init_form_left, init_form_right = st.columns((1, 1), gap='large')
            with init_form_left:
                st.subheader('密文图像')
                st.image('../data/images/plainimg.png', width=500)
            with init_form_right:
                st.subheader('解密图像')
                st.image('../data/images/cipherimg.png', width=500)
    if img_file:
        placeholder.empty()
        bytes_data = img_file.read()
        with open('../data/images/testdencry.png', 'wb+') as f:
            f.write(bytes_data)
        key = st.text_input(
            '输入图片提取码 👇 ',
            placeholder='请输入六位提取码',
        )
        img = Image.open('../data/images/testdencry.png')
        key_list = str.split(key, ' ')
        key_list = [int(val) for val in key_list if val != ' ' and val != '']
        print(key_list)
        with st.container():
            contact_form_left, contact_form_right = st.columns((1, 1), gap='small')
        with contact_form_left:
            st.subheader('密文图像')
            st.image(img)
        with contact_form_right:
            st.subheader('解密图像')
            if len(key_list) == 4:
                img = Decryption_img(r'../data/images/testdencry.png', key_list)
                st.image(img)
            else:
                pass
