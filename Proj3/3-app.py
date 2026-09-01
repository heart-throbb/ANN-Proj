import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


@st.cache_resource
def load_fashion_model():
    return load_model("model.keras")


model = load_fashion_model()


def preprocess_image(image):
    image = image.convert("L")
    image = image.resize((28, 28))
    image_array = np.array(image)
    image_array = image_array.astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


st.title("Fashion Item Classifier")
st.write(
    "Upload an image and the ANN will predict "
    "which Fashion-MNIST category it belongs to."
)
st.info(
    "For best results, upload a simple grayscale "
    "image with the clothing item centered."
)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.subheader("Uploaded Image")
    st.image(image, caption="Uploaded Image", width=300)
    if st.button("Predict"):
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image, verbose=0)[0]
        predicted_index = np.argmax(predictions)
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = predictions[predicted_index]

        st.success(f"Prediction: {predicted_class}")
        st.metric("Confidence", f"{confidence * 100:.2f}%")

        st.subheader("Top 3 Predictions")
        top_3_indices = np.argsort(predictions)[-3:][::-1]
        for rank, index in enumerate(top_3_indices, start=1):
            probability = predictions[index]
            st.write(f"**{rank}. {CLASS_NAMES[index]}**")
            st.progress(float(probability))
            st.caption(f"{probability * 100:.2f}%")
