import os
import sys
from pathlib import Path

import cv2
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import streamlit as st
from PIL import Image
# from HiNet.stegmodel import run_steg_model, init_steg_net
#
# steg_net = None
# if steg_net is None:
    # steg_net = init_steg_net()

def app():
    st.header('欢迎来到隐写模块')

    with st.container():
        contact_form_left, contact_form_right = st.columns((1, 1), gap='medium')
    with contact_form_left:
        img_file_cover = st.sidebar.file_uploader(label='上传一张需要封面图片', type=['png', 'jpg'], key='cover')
        if img_file_cover:
            st.subheader('封面图片')
            img_cover = Image.open(img_file_cover)
            st.image(img_cover)

    with contact_form_right:
        img_file_secret = st.sidebar.file_uploader(label='上传一张需要封面图片', type=['png', 'jpg'], key='secret')
        if img_file_secret:
            st.subheader('密文图片')
            img_secret = Image.open(img_file_secret)
            st.image(img_secret)
#
#     with st.container():
#         contact_form_next_left, contact_form_next_right = st.columns((1, 1), gap='medium')
#         if st.button('steg'):
#             run_steg_model(steg_net, img_cover, img_secret, 'steg')
#             st.write('steg done.')
