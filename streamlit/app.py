import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import streamlit as st
from multipage import MultiPage
from ltz import home, encrytion_page, steganography_page, decryption_page

st.set_page_config(page_title="图像加密系统", page_icon=":tiger:", layout="wide")

def local_css(file_name):
    with open(file_name, encoding='UTF-8') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("./style/style.css")
app = MultiPage()

# add applications
app.add_page('首页', home.app)
app.add_page('隐写', steganography_page.app)
app.add_page('加密', encrytion_page.app)
app.add_page('解密', decryption_page.app)


# Run application
if __name__ == '__main__':
    app.run()