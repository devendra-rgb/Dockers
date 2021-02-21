import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf


labels=['Melanocytic nevi',
'Melanoma',
'Benign keratosis',
'Basal cell carcinoma',
'Actinic keratoses',
'Vascular lesions',
'Dermatofibroma']

def get_output(img):
    model=tf.keras.models.load_model(r'cancer2.h5')
    #img=cv2.imread(img)
    #img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=np.resize(img,(28,28))
    img=img/255.0
    img=np.reshape(img,(-1,28,28,1))
    a=model.predict(img)
    ind=np.argmax(a)
    label=labels[ind]
    return label

st.title('Skin_Cancer_Detection')
st.write('By Devendra Tumu')
upload=st.file_uploader("Upload an jpg Image",type="jpg")
if upload is not None:
    image=Image.open(upload)
    st.image(image,caption="Uploaded Image",use_column_width=True)
    st.write(" ")
    st.write("Classifying",)
    label=get_output(image)
    st.header("We classified this as {}".format(label))
