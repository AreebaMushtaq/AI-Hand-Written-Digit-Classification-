import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import cv2

# Load trained model
model = tf.keras.models.load_model("digit_model.keras")

# Streamlit page setup
st.set_page_config(
    page_title="AI Digit Classifier",
    page_icon="✍️",
    layout="centered"
)

# Title
st.title("✍️ AI Handwritten Digit Recognition")

st.write("Draw a digit (0-9) below")

# Create drawing canvas
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Predict button
if st.button("Predict Digit"):

    if canvas_result.image_data is not None:

        # Get image from canvas
        img = canvas_result.image_data

        # Convert to grayscale
        img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGBA2GRAY)

        # Resize image to 28x28
        img = cv2.resize(img, (28, 28))

        # Normalize
        img = img / 255.0

        # Reshape for model
        img = img.reshape(1, 28, 28)

        # Prediction
        prediction = model.predict(img)

        predicted_digit = np.argmax(prediction)

        confidence = np.max(prediction) * 100

        # Show prediction
        st.success(f"Predicted Digit: {predicted_digit}")

        # Show confidence
        st.info(f"Confidence: {confidence:.2f}%")

        # Show probabilities
        st.subheader("Prediction Probabilities")

        probs = prediction[0]

        for i, prob in enumerate(probs):
            st.write(f"Digit {i}: {prob*100:.2f}%")
            st.progress(float(prob))