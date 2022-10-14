import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# pip install streamlit-cropper
import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image
from Encryption.EncryUtils import KeyGenerator
import os
# matplotlib==3.5.2


key = KeyGenerator('15461331')

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
add_selectbox = st.selectbox(
    '选择哪种模式进行加密?',
    ('整张加密', '自定义加密')
)

if img_file:
    if add_selectbox == '自定义加密':
        img = Image.open(img_file)
        # cv2.imwrite('./img.png', img)
        img.save('./img.png', quality=95)
        if not realtime_update:
            st.write("Double click to save crop")
        # Get a cropped image from the frontend
        cropped_img = st_cropper(img, realtime_update=realtime_update, box_color=box_color,
                                 aspect_ratio=aspect_ratio)
        cropped_img.save('./img1.png', quality=95)
        # Manipulate cropped image at will
        st.write("图片预览")
        _ = cropped_img.thumbnail((150, 150))
        st.image(cropped_img, channels='BGR', output_format='PNG')
    else:
        img = Image.open(img_file)
        st.image(img, channels='BGR', output_format='PNG')