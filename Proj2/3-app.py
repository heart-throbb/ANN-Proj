import pandas as pd
import pickle
import tensorflow as tf
import streamlit as st

model = tf.keras.models.load_model("model.h5")
with open("gender_encoder.pkl", "rb") as file:
    gender_encoder = pickle.load(file)
with open("geography_encoder.pkl", "rb") as file:
    geography_encoder = pickle.load(file)
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)
    
st.title("💰 Customer Salary Prediction")
st.write("Enter the customer's information below to predict their estimated salary.")

geography = st.selectbox("Geography", geography_encoder.categories_[0])
gender = st.selectbox("Gender", gender_encoder.classes_)
age = st.slider("Age", min_value=18, max_value=92, value=40)
balance = st.number_input("Balance", min_value=0.0, value=60000.0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
tenure = st.slider("Tenure", min_value=0, max_value=10, value=3)
num_of_products = st.slider("Number of Products", min_value=1, max_value=4, value=2)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
is_exited = st.selectbox("Is Exited", [0, 1])

if st.button("Predict Estimated Salary"):
    input_data = {
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_of_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active_member,
        "Exited": is_exited,
    }
    
    input_df = pd.DataFrame([input_data])
    
    input_df["Gender"] = gender_encoder.transform(input_df["Gender"])
    
    geo_encoded = geography_encoder.transform(input_df[["Geography"]]).toarray()
    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=geography_encoder.get_feature_names_out(["Geography"]),
        index=input_df.index,
    )
    input_df = pd.concat([input_df.drop("Geography", axis=1), geo_encoded_df], axis=1)
    
    expected_columns = [
        "CreditScore",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "Exited",
        "Geography_France",
        "Geography_Germany",
        "Geography_Spain",
    ]
    
    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)
    
    prediction = model.predict(scaled_input)
    predicted_salary = prediction[0][0]
    
    st.success(f"Predicted Estimated Salary: ${predicted_salary:,.2f}")
    st.write("### Customer Information")
    st.write(f"**Geography:** {geography}")
    st.write(f"**Gender:** {gender}")
    st.write(f"**Age:** {age}")
    st.write(f"**Credit Score:** {credit_score}")
    st.write(f"**Balance:** {balance:,.2f}")
