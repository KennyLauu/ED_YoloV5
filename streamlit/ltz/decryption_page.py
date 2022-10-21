import streamlit as st
from PIL import Image

def app():
    st.header('欢迎来到解密模块')
    img_file = st.sidebar.file_uploader(label='上传一张需要解密的图片', type=['png', 'jpg'])
    if img_file:
        img = Image.open(img_file)

        with st.container():
            contact_form_left, contact_form_right = st.columns((1, 1), gap='small')
        with contact_form_left:
            st.subheader('密文图像')
            st.image(img)
        with contact_form_right:
            st.subheader('解密图像')