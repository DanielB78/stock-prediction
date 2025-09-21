import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("meta_model.pkl")

st.title("Stock Movement Prediction")

uploaded_file = st.file_uploader("Upload CSV with features", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Input Data", df.head())

    preds = model.predict(df)
    df["Prediction"] = preds
    st.write("Results", df)