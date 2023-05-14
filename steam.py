import streamlit as st
from PIL import Image
import os
import cv2
from denoise_img import non_local_means_denoise


st.title("Welcome to Image Denoiser using NLM and CNN")

h = 0.5
sigma = 1.0
selected_option=''
def decideValues():
    if selected_option=="Low (Image Detailing will good)":
        return {'h':0.2,'sigma':1.2}
    elif selected_option=="Medium (Image Detailing will average)":
        return {'h':0.5,'sigma':1.2}
    else:
        return {'h':0.7,'sigma':1.0}
# Create a Streamlit file uploader widget
with st.sidebar:
   selected_option = st.radio("Quality of Denoising", ("Low (Image Detailing will good)", "Medium (Image Detailing will average)", "High (Image Detailing will less)"))
   st.write("You selected:", selected_option)
#    print(decideValues()['h'])
   uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
   i1 = st.button("Let's Generate Denoised Image")
   

# Check if a file was uploaded
if uploaded_file is not None and i1:
    # Get the path of the uploaded file

    image_path = os.path.join("./temp", uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # image_path = os.path.abspath(uploaded_file.name)

    # Print the image path
    # st.write("Image Path:", image_path)

    # Read the image file using PIL
    image = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(image, caption="Uploaded Image")

    input_image = cv2.imread(image_path)
    h=decideValues()['h']
    sigma=decideValues()['sigma']
    # print(h)
    # print(sigma)
    output_image = non_local_means_denoise(input_image, h=h, sigma=sigma)
    cv2.imwrite('outputs/output_image.jpg', output_image)

    st.image('outputs/output_image.jpg', caption="Denoised Image Using NLM with h 0.2")
    # output_image = non_local_means_denoise(input_image, h=0.5, sigma=1.2)
    # cv2.imwrite('outputs/output_image.jpg', output_image)

    # st.image('outputs/output_image.jpg', caption="Denoised Image Using NLM with h 0.5")
    # output_image = non_local_means_denoise(input_image, h=0.7, sigma=1.0)
    # cv2.imwrite('outputs/output_image.jpg', output_image)

    # st.image('outputs/output_image.jpg', caption="Denoised Image Using NLM with h 0.7")