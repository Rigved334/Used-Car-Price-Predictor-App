import joblib
import pandas as pd
import streamlit as st

model = joblib.load("Used_car_price.pkl")

st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗"
)

st.title("🚗 Used Car Price Predictor")

brand = st.text_input("Brand", "Toyota")
model_name = st.text_input("Model", "Camry")
year = st.number_input(
    "Model Year",
    1990,
    2026,
    2021
)

milage = st.number_input(
    "Mileage",
    0,
    500000,
    45000
)

fuel = st.selectbox(
    "Fuel Type",
    [
        "Gasoline",
        "Diesel",
        "Hybrid",
        "Electric",
        "Missing"
    ]
)

engine = st.text_input(
    "Engine",
    "2.5L I4"
)

transmission = st.selectbox(
    "Transmission",
    [
        "Automatic",
        "Manual"
    ]
)

ext_col = st.text_input(
    "Exterior Color",
    "White"
)

int_col = st.text_input(
    "Interior Color",
    "Black"
)

accident = st.selectbox(
    "Accident History",
    [
        "None reported",
        "At least 1 accident or damage reported"
    ]
)

clean_title = st.selectbox(
    "Clean Title",
    [
        "Yes",
        "No"
    ]
)

if st.button("Predict Price"):

    df = pd.DataFrame({

        "brand":[brand],
        "model":[model_name],
        "model_year":[year],
        "milage":[milage],
        "fuel_type":[fuel],
        "engine":[engine],
        "transmission":[transmission],
        "ext_col":[ext_col],
        "int_col":[int_col],
        "accident":[accident],
        "clean_title":[clean_title]

    })

    prediction = model.predict(df)[0]

    st.success(
        f"Estimated Price: ${prediction:,.2f}"
    )