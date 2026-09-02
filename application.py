import streamlit as st
import joblib

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "model (1).pkl")


st.title("Diabities detection application")

Pregnancies = st.number_input("Pregnancies", min_value=0.0, step = 1.0)
Glucose = st.number_input("Glucose", min_value=0.0)
BloodPressure = st.number_input("BloodPressure", min_value=0.0)
SkinThickness = st.number_input("SkinThickness", min_value=0.0)
Insulin = st.number_input("Insulin", min_value=0.0)
BMI = st.number_input("BMI", min_value=0.0)
DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction", min_value=0.0)
Age = st.number_input("Age", min_value=0.0, step = 1.0)

input_data = [
    Pregnancies,
    Glucose,
    BloodPressure,
    SkinThickness,
    Insulin,
    BMI,
    DiabetesPedigreeFunction,
    Age,
]

if st.button("Predict"):
    prob = model.predict_proba([input_data])[0][1]
    result = "1 (Positive)" if prob > 0.4 else "0 (Negative)"
    st.write("Prediction:", result)
    st.write("Diabetes Probability:", round(prob, 4))
