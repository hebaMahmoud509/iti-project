

# import needed libraries
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# road segmentation using Kmeans
def segmentation(img):
  grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  pixel_values = grayscale.reshape((-1,3))
  pixel_values = np.float32(pixel_values)
  criteria = (cv2.TERM_CRITERIA_EPS+ cv2.TERM_CRITERIA_MAX_ITER,10,1)
  _, labels,(centers) = cv2.kmeans(pixel_values,3,None, criteria,10,cv2.KMEANS_RANDOM_CENTERS)
  centers = np.uint8(centers)
  labels = centers[labels.flatten()]
  return labels.reshape((grayscale.shape));

#sing the file uploader module provided by streamlit to upload images while specifying the extensions to support different formats
uploaded_file = st.file_uploader("", type=['jpg', 'png', 'jpeg'])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = np.array(image.convert('RGB'))
    segmented_img = segmentation(img)
    if segmented_img is not None:
        st.image(cv2.cvtColor(segmented_img, cv2.COLOR_GRAY2BGR))