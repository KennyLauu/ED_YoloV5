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
    img = plt.imread(img_file)
    plt.imshow(img)
    plt.show()
    # img = cv2.imread('../data/images/bus.jpg')
    # cv2.imshow('title', img)
    # cv2.waitKey(0)

    # 私钥
    key = ProcessingKey(img)
    keyd = [39, 16, 78, 6]
    keyc = [39, 16, 77, 7]

    # 加密图像
    print('加密')
    EncryImg = noColorEncry(img, key)
    plt.imshow(EncryImg)
    plt.title('encryption')
    plt.show()

    # 解密图像
    print('解密')
    DecryImg = noColorDecry.noColorDecry(EncryImg, key)
    plt.imshow(DecryImg)
    plt.title('decryption')
    plt.show()

    # 比较原图与解密图的差异
    gap = np.abs(DecryImg - img)
    plt.imshow(gap)
    plt.show()
    print('差异', max(gap.flatten()))




    # img = Image.open(img_file)
    # if not realtime_update:
    #     st.write("点击两下保存")
    # # Get a cropped image from the frontend
    # cropped_img = st_cropper(img, realtime_update=realtime_update, box_color=box_color,
    #                          aspect_ratio=aspect_ratio, return_type='box')
    # print(cropped_img)
    # # Manipulate cropped image at will
    # st.write("预览截图")
    # # _ = cropped_img.thumbnail((150, 150))
    # # st.image(cropped_img)
