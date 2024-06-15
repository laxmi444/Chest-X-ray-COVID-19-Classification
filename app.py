import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the trained model
model_path = 'models/xray_model.h5'
st.write(f"Loading model from: {model_path}")
try:
    model = tf.keras.models.load_model(model_path)
    st.write("Model loaded successfully.")
except Exception as e:
    st.write(f"Error loading model: {e}")

# Function to preprocess the uploaded image
def preprocess_image(image):
    try:
        # Convert the image to RGB if required
        image = image.convert('RGB')
        # Resize the image to the input size required by your model
        image = image.resize((100, 100))
        # Convert image to numpy array
        image = np.array(image)
        # Normalize the image
        image = image / 255.0
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        st.write(f"Error in image preprocessing: {e}")
        return None

# Streamlit app
st.title("Chest X-ray COVID-19 Classifier🩻")
st.write("Upload a chest X-ray image to classify whether it indicates COVID-19 or not.")

# File uploader allows user to upload an image
uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "png"])

if uploaded_file is not None:
    try:
        # Load the image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded X-ray.', use_column_width=True)

        # Preprocess the image
        processed_image = preprocess_image(image)
        if processed_image is not None:
            # Make prediction
            prediction = model.predict(processed_image)[0][0]  # Get the first and only prediction

            # Display the prediction
            if prediction > 0.5:
                st.write("The model predicts: **COVID-19**")
            else:
                st.write("The model predicts: **No COVID-19**")
    except Exception as e:
        st.write(f"Error during prediction: {e}")
else:
    st.write("Please upload an image.")

