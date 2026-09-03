import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model

MODEL_PATH = "model.keras"
SCALER_PATH = "scaler.pkl"
DATASET_PATH = "Dataset/creditcard.csv"
THRESHOLD = 0.5

st.title("Credit Card Fraud Detection")

if not os.path.exists(MODEL_PATH):
    st.error("model.keras was not found. " "Train the model first.")
    st.stop()

if not os.path.exists(SCALER_PATH):
    st.error("scaler.pkl was not found. " "Train the model first.")
    st.stop()


if not os.path.exists(DATASET_PATH):
    st.error("Dataset/creditcard.csv was not found.")
    st.stop()


@st.cache_resource
def load_fraud_model():
    return load_model(MODEL_PATH)


@st.cache_resource
def load_scaler():
    with open(SCALER_PATH, "rb") as file:
        return pickle.load(file)


model = load_fraud_model()
scaler = load_scaler()

data = pd.read_csv(DATASET_PATH, nrows=1)
feature_names = [column for column in data.columns if column != "Class"]

st.subheader("Transaction Information")
st.write("Enter the transaction feature values below.")

left_column, right_column = st.columns(2)
feature_values = {}

for index, feature in enumerate(feature_names):
    if index % 2 == 0:
        with left_column:
            feature_values[feature] = st.number_input(
                label=feature, value=0.0, format="%.6f"
            )
    else:
        with right_column:
            feature_values[feature] = st.number_input(
                label=feature, value=0.0, format="%.6f"
            )

st.divider()
if st.button("Analyze Transaction", use_container_width=True):
    input_data = pd.DataFrame([feature_values], columns=feature_names)
    input_scaled = scaler.transform(input_data)
    probability = model.predict(input_scaled, verbose=0)[0][0]
    prediction = int(probability >= THRESHOLD)
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error("FRAUD detected!!!!!")
    else:
        st.success("NORMAL....")
    st.metric("Fraud Probability", f"{probability * 100:.4f}%")
    st.progress(float(probability))
