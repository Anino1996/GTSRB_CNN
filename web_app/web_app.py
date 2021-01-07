import numpy as np 
import pandas as pd 
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import os
import json

@st.cache
def load_classes():
	with open('sign_classes.json','r') as file:
		class_names=json.load(file)
	return class_names


#Load model and class names
model=load_model('final_model.h5')
sign_classes=load_classes()


#Change Image Dimensions
def dim_change(file_path):
  im=Image.open(file_path)
  im=im.resize((32,32))
  data=np.array(im).astype('float32')
  return data/255.0

st.markdown("<h1 style= 'text-align:center'> EEN-614 Neural Networks Final Project </h1>", unsafe_allow_html=True)
st.markdown("<h2 style = 'text-align: center'>TRAFFIC SIGN CLASSIFICATION SYSTEM USING CNNs</h2><p style = 'text-align: center'><b>Albert Aninagyei Ofori - Norfolk State University</b></p>", unsafe_allow_html=True)
img=st.file_uploader("", type=["jpg", "jpeg", "png", "bmp"])

if img==None:
	st.markdown("<p style='color: grey; text-align: center; font-size: 40px'> Please Upload Your Image <span style='font-size:20px'> (png, jpg, bmp or jpeg)</span></p>", unsafe_allow_html=True)

else:
	try:
		img_data=np.array([dim_change(img)])
		image_prediction=model.predict(img_data)[0]
		image_class=sign_classes[str(np.argmax(image_prediction))]

		st.markdown(f"<h3 style= 'text-align:center'> This image is a <b>{image_class}</b> sign </h3>", unsafe_allow_html=True)
		st.image(img,use_column_width=True)

	except:
		st.markdown("<p style='color: grey; text-align: center; font-size: 30px'> Image format error. Please use another image </p>", unsafe_allow_html=True)

st.markdown("<h3 style= 'text-align:center'> 43 Road Signs Supported in this project </h3>", unsafe_allow_html=True)
st.table(pd.Series(data=list(sign_classes.values()),name="Signs", index=range(1,len(sign_classes)+1)))





