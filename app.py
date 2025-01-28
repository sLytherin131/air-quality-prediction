import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Load saved XGBoost model from local directory
model_path = os.path.join(os.getcwd(), 'xgboost_model.pkl')
loaded_xgb_model = joblib.load(model_path)

# Load saved scaler from local directory
scaler_path = os.path.join(os.getcwd(), 'scaler.pkl')
scaler = joblib.load(scaler_path)

# Load saved LabelEncoder from local directory
label_encoder_path = os.path.join(os.getcwd(), 'label_encoder.pkl')
label_encoder = joblib.load(label_encoder_path)

# Streamlit App Title
st.title("Air Quality Prediction")

# User Inputs
pm25 = st.number_input("Masukkan nilai PM2.5:", min_value=0.0, step=0.1)
so2 = st.number_input("Masukkan nilai SO2:", min_value=0.0, step=0.1)
pm10 = st.number_input("Masukkan nilai PM10:", min_value=0.0, step=0.1)
co = st.number_input("Masukkan nilai CO:", min_value=0.0, step=0.1)
o3 = st.number_input("Masukkan nilai O3:", min_value=0.0, step=0.1)
no2 = st.number_input("Masukkan nilai NO2:", min_value=0.0, step=0.1)

# Prediction button
if st.button('Prediksi Kualitas Udara'):
    # Create a DataFrame for the input data
    input_data = pd.DataFrame({
        'PM10': [pm10],
        'PM25': [pm25],
        'SO2': [so2],
        'CO': [co],
        'O3': [o3],
        'NO2': [no2]
    })

    # Apply the same scaling used during training
    input_data_scaled = scaler.transform(input_data)

    # Make prediction using the loaded model
    prediction = loaded_xgb_model.predict(input_data_scaled)

    # Convert prediction back to category using LabelEncoder
    predicted_category = label_encoder.inverse_transform(prediction)

    # Display predicted category
    st.write(f"Kategori kualitas udara yang diprediksi: {predicted_category[0]}")
