import streamlit as st
import pandas as pd
import numpy as np
import joblib

# PAGE CONFIGURATION

st.set_page_config(
    page_title="Crop Yield Prediction System",
    page_icon="🌾",
    layout="centered"
)

x = pd.read_csv("C:\\Users\\HP\\Downloads\\x_train_crop_prediction.csv")

feature_columns = x.columns.tolist()

# GET CROP AND COUNTRY OPTIONS

# Crop columns: index 4 to 13
crop_options = x.columns[4:14].tolist()

# Country columns: index 14 to 114
country_options = x.columns[14:115].tolist()

# LOAD TRAINED MODEL

model = joblib.load("C:\\Users\\HP\\Downloads\\model_crop_prediction.joblib")

# APPLICATION TITLE

st.title("🌾 Crop Yield Prediction System")

st.write(
    "Enter the agricultural and environmental information "
    "below to predict crop yield."
)

st.divider()
# USER INPUTS
st.subheader("Enter Crop Information")

# YEAR

year = st.number_input(
    "Year",
    min_value=1990,
    max_value=2030,
    value=2020,
    step=1
)

rainfall = st.number_input(
    "Average Annual Rainfall (mm/year)",
    min_value=0.0,
    value=1000.0,
    step=10.0
)

# PESTICIDE USAGE

pesticides = st.number_input(
    "Pesticide Usage (tonnes)",
    min_value=0.0,
    value=100.0,
    step=1.0
)

# AVERAGE TEMPERATURE

temperature = st.number_input(
    "Average Temperature (°C)",
    min_value=-10.0,
    max_value=60.0,
    value=25.0,
    step=0.1
)

# CROP

crop = st.selectbox(
    "Crop",
    crop_options
)

# COUNTRY
country = st.selectbox(
    "Country",
    country_options
)
st.divider()
scaler = joblib.load("C:\\Users\\hp\\Downloads\\scaler_crop_prediction.joblib")

# PREDICTION

if st.button("🌱 Predict Crop Yield", use_container_width=True):
    numerical_data = np.array([
    [year, rainfall, pesticides, temperature]
   ])
    scaled_data = scaler.transform(numerical_data)

    input_data = pd.DataFrame(
        0.0,
        index=[0],
        columns=feature_columns
    )
    input_data.iloc[0, 0] = scaled_data[0, 0]
    input_data.iloc[0, 1] = scaled_data[0, 1]
    input_data.iloc[0, 2] = scaled_data[0, 2]
    input_data.iloc[0, 3] = scaled_data[0, 3]
    input_data.loc[0, crop] = 1
    input_data.loc[0, country] = 1

    prediction = model.predict(input_data)

    predicted_yield = prediction[0]

    st.success("Prediction completed successfully!")

    st.subheader("Predicted Crop Yield")

    st.metric(
        label="Predicted Yield",
        value=f"{predicted_yield:,.2f}"
    )
    st.subheader("Prediction Details")

    result = pd.DataFrame({
        "Feature": [
            "Year",
            "Average Annual Rainfall",
            "Pesticide Usage",
            "Average Temperature",
            "Crop",
            "Country"
        ],

        "Value": [
            year,
            f"{rainfall:,.2f} mm/year",
            f"{pesticides:,.2f} tonnes",
            f"{temperature:.2f} °C",
            crop,
            country
        ]
    })
    st.table(result)