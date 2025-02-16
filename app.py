import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


tf.compat.v1.reset_default_graph()  # Instead of tf.reset_default_graph()

# Load the trained LSTM model
# model = tf.keras.models.load_model("model.h5")

from tensorflow.keras.models import load_model

model = load_model("model.h5", safe_mode=False)


# Manually compile the model with metrics
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Load the saved feature & target scalers
scaler_X = joblib.load("scaler_X.pkl")  # For input features
scaler_y = joblib.load("scaler_y.pkl")  # For output target

def main():
    # Streamlit App Interface
    st.title("🌦️ LSTM Weather Forecaster")

    st.markdown("Enter weather conditions to predict if it will rain.")

    # User input fields
    temperature = st.number_input("Temperature (°C)", min_value=-30.0, max_value=50.0, value=25.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
    wind_speed = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=150.0, value=10.0)
    pressure = st.number_input("Pressure (hPa)", min_value=800.0, max_value=1100.0, value=1013.0)
    cloud_cover = st.number_input("Cloud Cover (%)", min_value=0.0, max_value=100.0, value=50.0)

    # Make a prediction when the button is clicked
    if st.button("Predict"):
        # Convert user input into an array
        input_data = np.array([[temperature, humidity, wind_speed, pressure, cloud_cover]])

        # Apply the feature scaler
        input_scaled = scaler_X.transform(input_data)  # Scale input like training data

        # Reshape input for LSTM (samples, time_steps, features)
        input_reshaped = input_scaled.reshape((1, 1, input_scaled.shape[1]))

        # Make prediction
        prediction_scaled = model.predict(input_reshaped)

        # Inverse transform the prediction to get actual scale
        prediction_actual = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))

        # Convert to binary classification
        result = "🌧️ Rain Expected" if prediction_actual[0][0] > 0.5 else "☀️ No Rain Expected"

        # Display result
        st.subheader(f"Prediction: {result}")
        st.write(f"Rain Probability: {prediction_actual[0][0]:.2%}")

if __name__ == '__main__':
    main()
