import streamlit as st
import requests
from PIL import Image


# Use local CSS
def local_css(file_name):
    with open(file_name, encoding='UTF-8') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def app():
    local_css("./style/style.css")
    # ---- LOAD ASSETS ----
    allEncry_img = Image.open("images/allEncry_img.jpg")
    YoloEncry_img = Image.open("images/YoloEncry_img.png")
    Yoloseg_img = Image.open("images/Yolo_seg_img.png")

    # ---- HEADER SECTION ----

    with st.container():
        st.subheader("欢迎来到基于YoloV5+DeepSort的智能影像加密系统")
        banner_img = """
                        <div id="banner"></div>
                """
        st.markdown(banner_img, unsafe_allow_html=True)
        # st.title("")
        # st.write(
        #     "I'm passionate on fusing polymer physics and artificial intelligence technology.."
        # )
        # st.write("[Learn More >](https://space.bilibili.com/76811961)")



    # with st.container():
    #     contact_form_left = """
    #                 <button type="submit">加密</button>
    #             """
    #     contact_form_right = """
    #                 <button type="submit">解密</button>
    #             """
    #     left_column, right_column = st.columns(2)
    #     with left_column:
    #         st.markdown(contact_form_left, unsafe_allow_html=True)
    #     with right_column:
    #         st.markdown(contact_form_right, unsafe_allow_html=True)

    # ---- WHAT I DO ----
    with st.container():
        st.write("---")
        left_column, right_column = st.columns(2)
        with left_column:
            st.header("关于我们的功能")
            st.write("##")
            st.write(
                """
                我们的系统是针对互联网安全进行的一个图像加密或者解密，主要功能如下
                - 对图片进行自定义加密解密
                - 基于实例分割进行的加密解密 
                """
            )
            # st.write("[Bilibili Channel >](https://space.bilibili.com/76811961)")
        with right_column:
            st.image('images/sphere.jpg')

    st.write("---")
    st.header("用例如下")
    st.write("##")

    with st.container():
        text_column, image_column = st.columns((2, 3), gap='medium')
        with image_column:
            st.image(allEncry_img)
        with text_column:
            st.subheader("自定义加密")
            st.write(
                """
                此功能可以对图像进行自定义加密
                """
            )
        with st.expander('详解请点击此处'):
            st.write('首先上传一张图片，然后我们会通过目标检测的方式将常规物体的检测出来，然后再通过图像加密的方式，只对用户感兴趣的部分进行加密')

    st.write('##')
    st.write("---")

    # with st.container():
    #     text_column, image_column = st.columns((2, 3), gap='medium')
    #     with image_column:
    #         st.image(YoloEncry_img)
    #
    #     with text_column:
    #         st.subheader("基于目标检测的图像加密")
    #         st.write(
    #             """
    #             此功能是基于目标检测的图像加密，首先选择好检测出来的目标，然后对选取目标进行加密
    #             """
    #         )
    #         # st.markdown("[Watch Video...](https://www.bilibili.com/video/BV1TT4y1J72n)")
    #     with st.expander('详解请点击此处'):
    #         st.write('首先上传一张图片，然后我们会通过目标检测的方式将常规物体的检测出来，然后再通过图像加密的方式，只对用户感兴趣的部分进行加密')
    #
    # st.write('##')
    # st.write("---")

    with st.container():
        text_column, image_column = st.columns((2, 5), gap='medium')
        with image_column:
            st.image(Yoloseg_img)
        with text_column:
            st.subheader("基于实例分割的图像加密")
            st.write(
                """
                此功能是实例分割的图像加密，首先选择好检测出来的目标，然后对选取目标进行加密
                """
            )
            # st.markdown("[Watch Video...](https://www.bilibili.com/video/BV1yt4y1277N)")
            st.write(' ')
            st.write('')
        with st.expander('详解请点击此处'):
            st.write('首先上传一张图片，然后我们会通过目标检测的方式将常规物体的检测出来，然后再通过图像加密的方式，只对用户感兴趣的部分进行加密')

    st.write('##')
    st.write("---")

    # # ---- CONTACT ----
    # with st.container():
    #     st.write("---")
    #     st.header("Get In Touch With Me!")
    #     st.write("##")
    #
    #     # Documention: https://formsubmit.co/ !!! CHANGE EMAIL ADDRESS !!!
    #     contact_form = """
    #     <form action="https://formsubmit.co/YOUR@MAIL.COM" method="POST">
    #         <input type="hidden" name="_captcha" value="false">
    #         <input type="text" name="name" placeholder="Your name" required>
    #         <input type="email" name="email" placeholder="Your email" required>
    #         <textarea name="message" placeholder="Your message here" required></textarea>
    #         <button type="submit">Send</button>
    #     </form>
    #     """
    #     left_column, right_column = st.columns(2)
    #     with left_column:
    #         st.markdown(contact_form, unsafe_allow_html=True)
    #     with right_column:
    #         st.empty()
