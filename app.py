import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Title
st.title("♻️ Trash Classification App")
st.write("Upload a trash image and let the model predict its category.")

# Load model once
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("trashnet_model.h5")
    return model

model = load_model()

# Class labels (adjust if your dataset is different)
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Upload image
uploaded_file = st.file_uploader("Upload an image of trash...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))  # Adjust if your model uses different size
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array / 255.0  # Normalize if model trained on normalized data
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension

    # Prediction
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])  # If last layer is not softmax, you can use argmax directly

    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    # Display prediction
    st.markdown(f"### 🧠 Prediction: **{predicted_class}**")
    st.markdown(f"Confidence: **{confidence:.2f}%**")

