import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("salary_model.joblib")

# Website title
st.title("Salary Prediction App")

st.write("Predict salary using work experience.")

# Get experience from the user
experience = st.number_input(
    "Work experience (months)",
    min_value=0.0,
    max_value=100.0,
    value=24.0
)

# Predict salary
if st.button("Predict Salary"):

    input_data = pd.DataFrame({
        "exp(in months)": [experience]
    })

    prediction = model.predict(input_data)[0]

    st.success(
        f"Estimated salary: {prediction:.2f} thousand"
    )