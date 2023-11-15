import numpy as np
from keras.models import load_model
from PIL import Image
import streamlit as st

def preprocess_image(image):
    image = image.resize((INPUT_SIZE, INPUT_SIZE))
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

INPUT_SIZE = 64

model = load_model('BrainTumor10Epochs.h5')

st.title('Brain Tumor Detection')

uploaded_file = st.file_uploader('Choose an image of a brain with a tumor', type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)

        st.header('Original Image')
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Add "Process Image" button and make it clickable
        if st.button("Process Image"):
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            if prediction[0][0] > 0.5:
                result = 'Tumor detected.'
            else:
                result = 'No tumor detected.'

            st.header('Result')
            st.write('Prediction:', result)

    except Exception as e:
        st.error("Error: Unable to process the image. Please make sure it's a valid image file.")
