import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3

from pathlib import Path
from datetime import datetime

# PAGE CONFIGURATION

st.set_page_config(
    page_title="Crop Yield Prediction System",
    page_icon="🌾",
    layout="centered"
)

# DATABASE CONFIGURATION

# Database file will be created beside this Python file
DB_PATH = Path(__file__).resolve().parent / "crop_yield_predictions.db"
def initialize_database():
    """Create the predictions table if it does not exist."""
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_time TEXT NOT NULL,
                year INTEGER NOT NULL,
                rainfall REAL NOT NULL,
                pesticides REAL NOT NULL,
                temperature REAL NOT NULL,
                crop TEXT NOT NULL,
                country TEXT NOT NULL,
                predicted_yield REAL NOT NULL
            )
        """)
        conn.commit()
    finally:
        conn.close()

def save_prediction(
    year,
    rainfall,
    pesticides,
    temperature,
    crop,
    country,
    predicted_yield
):
    """Save a prediction and its inputs to the database."""
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("""
            INSERT INTO predictions (
                prediction_time,
                year,
                rainfall,
                pesticides,
                temperature,
                crop,
                country,
                predicted_yield
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().astimezone().isoformat(timespec="seconds"),
            int(year),
            float(rainfall),
            float(pesticides),
            float(temperature),
            crop,
            country,
            float(predicted_yield)
        ))
        conn.commit()
    finally:
        conn.close()
def get_prediction_history():
    """Retrieve saved predictions, newest first."""
    conn = sqlite3.connect(DB_PATH)
    try:
        history = pd.read_sql_query("""
            SELECT
                id AS 'Prediction ID',
                prediction_time AS 'Date and Time',
                year AS 'Year',
                rainfall AS 'Rainfall (mm/year)',
                pesticides AS 'Pesticide Usage (tonnes)',
                temperature AS 'Temperature (°C)',
                crop AS 'Crop',
                country AS 'Country',
                predicted_yield AS 'Predicted Yield'
            FROM predictions
            ORDER BY id DESC
        """, conn)

    finally:
        conn.close()
    return history
# Initialize database automatically
initialize_database()
# LOAD DATA AND TRAINED MODEL

x = pd.read_csv("x_train_crop_prediction.csv")

feature_columns = x.columns.tolist()
# Crop columns: index 4 to 13
crop_options = x.columns[4:14].tolist()
# Country columns: index 14 to 114
country_options = x.columns[14:115].tolist()
model = joblib.load("model_crop_prediction.joblib")
scaler = joblib.load("scaler_crop_prediction.joblib")
# APPLICATION TITLE
st.title("🌾 Crop Yield Prediction System")
st.write(
    "Enter the agricultural and environmental information "
    "below to predict crop yield."
)
st.divider()

# USER INPUTS
st.subheader("Enter Crop Information")
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

pesticides = st.number_input(
    "Pesticide Usage (tonnes)",
    min_value=0.0,
    value=100.0,
    step=1.0
)

temperature = st.number_input(
    "Average Temperature (°C)",
    min_value=-10.0,
    max_value=60.0,
    value=25.0,
    step=0.1
)

crop = st.selectbox(
    "Crop",
    crop_options
)

country = st.selectbox(
    "Country",
    country_options
)

st.divider()
# PREDICTION
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if st.button(
    "🌱 Predict Crop Yield",
    use_container_width=True
):
    # Scale numerical input
    numerical_data = np.array([[
        year,
        rainfall,
        pesticides,
        temperature
    ]])
    scaled_data = scaler.transform(numerical_data)
    # Create model input with all expected features
    input_data = pd.DataFrame(
        0.0,
        index=[0],
        columns=feature_columns
    )
    input_data.iloc[0, 0] = scaled_data[0, 0]
    input_data.iloc[0, 1] = scaled_data[0, 1]
    input_data.iloc[0, 2] = scaled_data[0, 2]
    input_data.iloc[0, 3] = scaled_data[0, 3]
    # Set crop and country indicators
    input_data.loc[0, crop] = 1
    input_data.loc[0, country] = 1
    # Generate prediction
    prediction = model.predict(input_data)
    predicted_yield = float(prediction[0])
    # Save to SQLite database
    save_prediction(
        year=year,
        rainfall=rainfall,
        pesticides=pesticides,
        temperature=temperature,
        crop=crop,
        country=country,
        predicted_yield=predicted_yield
    )
    # Keep the latest result visible after Streamlit reruns
    st.session_state.last_prediction = {
        "year": year,
        "rainfall": rainfall,
        "pesticides": pesticides,
        "temperature": temperature,
        "crop": crop,
        "country": country,
        "predicted_yield": predicted_yield
    }
    st.success("Prediction completed and saved to the database!")
# DISPLAY LATEST PREDICTION
if st.session_state.last_prediction is not None:
    result_data = st.session_state.last_prediction
    st.subheader("Predicted Crop Yield")
    st.metric(
        label="Predicted Yield",
        value=f"{result_data['predicted_yield']:,.2f}"
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
            result_data["year"],
            f"{result_data['rainfall']:,.2f} mm/year",
            f"{result_data['pesticides']:,.2f} tonnes",
            f"{result_data['temperature']:.2f} °C",
            result_data["crop"],
            result_data["country"]
        ]
    })
    st.table(result)
# DATABASE: PREDICTION HISTORY
st.divider()
st.subheader("📚 Prediction History")
st.write(
    "All previously saved crop yield predictions "
    "are displayed below."
)
history = get_prediction_history()
if history.empty:
    st.info(
        "No predictions have been saved yet. "
        "Make your first prediction to create a history record."
    )
else:
    st.dataframe(
        history,
        use_container_width=True,
        hide_index=True
    )
    st.caption(
        f"Total predictions saved: {len(history)}"
    )
