import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps


model = tf.keras.models.load_model("D:\Machine learning\handwrittennumber.h5")  

st.title("Handwritten Digit Recognition")
st.write("Upload an image of a handwritten digit (0-9) to get the prediction.")


uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
   
    image = image.resize((28, 28))  
    image = ImageOps.invert(image)   
    img_array = np.array(image) / 255.0  

    img_array = img_array.reshape(1, 28*28)  


    img_array = img_array.reshape(1, 28, 28, 1)  
    
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)
    
    st.write(f"Predicted Digit: **{predicted_digit}**")
