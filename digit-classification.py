from keras.saving.save import save_model
import streamlit as st
import numpy as np
import pandas as pd
import os
from skimage.color import rgb2lab, lab2rgb
from skimage.io import imsave, imshow
from skimage.transform import resize
import time
from keras.datasets import mnist
from keras import Sequential
from keras.models import load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Input
from keras.preprocessing.image import load_img
from PIL import Image

st.title("Digit Classification")

model = load_model("digit-model.h5")

# Layout
sample_img =  st.checkbox("Check for sample images")
image = st.file_uploader("Upload the image of digit", type=["png", "jpg"])

if sample_img:
    images = st.sidebar.selectbox("Select a sample image", os.listdir("Digit Samples"))

    if images is not None:
        img_sam = load_img(os.path.join("Digit Samples", images), color_mode="grayscale")
        c1, c2, c3 = st.beta_columns(3)
        c2.image(img_sam, width=150)
        c1, c2, c3 = st.beta_columns(3)
        if c2.button("Classify"):

            img = np.expand_dims(img_sam, (0, -1))
            pred = model.predict_classes(img)[0]

            st.success(f"The Image looks like a **{pred}**")
        
if image is not None:
    try:
        image = Image.open(image)
        image.save("image.png")

        pred_img = load_img("image.png", color_mode="grayscale")
        c1, c2, c3 = st.beta_columns(3)
        c2.image(pred_img, width=150)
        c1, c2, c3 = st.beta_columns(3)
        if c2.button("Classify"):

            img = np.expand_dims(img_sam, (0, -1))
            pred = model.predict_classes(img)[0]

            st.success(f"The Image looks like a **{pred}**")

    except Exception as e:
        er = st.error("Image cannot be processed!")