# app/streamlit_app.py 
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ==================== LOAD MODEL ====================
@st.cache_resource
def load_model():
    return joblib.load("car_price_prediction/models/model.pkl")

pipeline = load_model()
preprocessor = pipeline['preprocessor']
model = pipeline['model']

# ==================== PAGE CONFIG ====================
st.set_page_config(page_title="Pakistani Car Price Predictor", page_icon="Pakistan", layout="centered")

# ==================== HEADER ====================
st.markdown("""
    <h1 style='text-align: center; color: #FF4B4B; margin-bottom: 8px;'>
        Pakistani Car Price Predictor
    </h1>
    <p style='text-align: center; color: #666; font-size: 18px;'>
        Based on 5,000+ verified listings
    </p>
    <hr style='border: 2px solid #FF4B4B; border-radius: 5px;'>
""", unsafe_allow_html=True)

# ==================== MODEL LIST  ====================
brand_to_models = {
    "Toyota": ["Corolla", "Yaris", "Camry", "Prius", "Aqua", "Fortuner", "Revo", "Hilux", "Altis Grande"],
    "Honda": ["Civic", "City", "BR-V", "HR-V", "Vezel", "Accord", "CR-V"],
    "Suzuki": ["Cultus", "Wagon R", "Swift", "Alto", "Mehran", "Bolan", "Ciaz"],
    "Kia": ["Sportage", "Picanto", "Sorento", "Stonic", "Carnival"],
    "Hyundai": ["Tucson", "Elantra", "Sonata", "Santa Fe", "H-1"],
    "Mercedes": ["C-Class", "E-Class", "S-Class", "GLC", "GLA", "GLE"],
    "BMW": ["3 Series", "5 Series", "7 Series", "X1", "X3", "X5", "X7"],
    "Audi": ["A3", "A4", "A6", "A8", "Q3", "Q5", "Q7", "Q8"],
    "MG": ["HS", "ZS", "ZS EV", "RX8"],
    "Changan": ["Alsvin", "Karry", "Oshan X7"],
    "Haval": ["H6", "Jolion", "H9"],
    "Other": ["Any Other Model"]
}

# ==================== SMART MODEL SPECS ====================
model_specs = {
    "Prius":     {"fuel": ["Hybrid"],           "transmission": ["Automatic"]},
    "Aqua":      {"fuel": ["Hybrid"],           "transmission": ["Automatic"]},
    "Vezel":     {"fuel": ["Hybrid", "Petrol"], "transmission": ["Automatic"]},
    "HR-V":      {"fuel": ["Petrol"],           "transmission": ["Automatic", "Manual"]},
    "Civic":     {"fuel": ["Petrol"],           "transmission": ["Automatic", "Manual"]},
    "City":      {"fuel": ["Petrol"],           "transmission": ["Automatic", "Manual"]},
    "Corolla":   {"fuel": ["Petrol"],           "transmission": ["Automatic", "Manual"]},
    "Yaris":     {"fuel": ["Petrol"],           "transmission": ["Automatic", "Manual"]},
    "Alto":      {"fuel": ["Petrol"],           "transmission": ["Manual", "Automatic"]},
    "Cultus":    {"fuel": ["Petrol", "CNG"],    "transmission": ["Manual"]},
    "Wagon R":   {"fuel": ["Petrol"],           "transmission": ["Manual", "Automatic"]},
    "Mehran":    {"fuel": ["Petrol", "CNG"],    "transmission": ["Manual"]},
    "Sportage":  {"fuel": ["Petrol", "Diesel"], "transmission": ["Automatic"]},
    "Tucson":    {"fuel": ["Petrol", "Diesel"], "transmission": ["Automatic"]},
    "Fortuner":  {"fuel": ["Diesel"],           "transmission": ["Automatic", "Manual"]},
    "Revo":      {"fuel": ["Diesel"],           "transmission": ["Automatic", "Manual"]},
    "Hilux":     {"fuel": ["Diesel"],           "transmission": ["Automatic", "Manual"]},
    "Picanto":   {"fuel": ["Petrol"],           "transmission": ["Manual", "Automatic"]},
    "Stonic":    {"fuel": ["Petrol"],           "transmission": ["Automatic"]},
    "Elantra":   {"fuel": ["Petrol"],           "transmission": ["Automatic"]},
    "Sonata":    {"fuel": ["Petrol"],           "transmission": ["Automatic"]},
    "C-Class":   {"fuel": ["Petrol", "Diesel"], "transmission": ["Automatic"]},
    "E-Class":   {"fuel": ["Petrol", "Diesel"], "transmission": ["Automatic"]},
    "A3":        {"fuel": ["Petrol", "Hybrid"], "transmission": ["Automatic"]},
    "Q5":        {"fuel": ["Petrol", "Hybrid"], "transmission": ["Automatic"]},
    "H6":        {"fuel": ["Petrol"],           "transmission": ["Automatic"]},
    "Jolion":    {"fuel": ["Petrol", "Hybrid"], "transmission": ["Automatic"]},
    "HS":        {"fuel": ["Petrol"],           "transmission": ["Automatic"]},
    "ZS":        {"fuel": ["Petrol"],           "transmission": ["Manual", "Automatic"]},
}

def get_fuel_options(model_name):
    return model_specs.get(model_name, {}).get("fuel", ["Petrol", "Hybrid", "Diesel", "CNG", "Electric"])

def get_transmission_options(model_name):
    return model_specs.get(model_name, {}).get("transmission", ["Automatic", "Manual"])

# ==================== INPUT FORM ====================
st.markdown("#### Enter Car Details")
col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", options=sorted(brand_to_models.keys()))
    model_input = st.selectbox("Model", options=brand_to_models[brand])
    condition = st.selectbox("Condition", ["Used", "Imported", "New"])

    if condition == "New":
        year = 2025
        st.success("New car → Year set to **2025**")
        st.info("Only current-year models are considered 'New'")
    else:
        year = st.slider("Manufacturing Year", 2000, 2025, 2022, format="%d")

with col2:
    fuel = st.selectbox("Fuel Type", options=get_fuel_options(model_input))
    transmission = st.selectbox("Transmission", options=get_transmission_options(model_input))
    
    city = st.selectbox("City", ["Lahore", "Karachi", "Islamabad", "Rawalpindi", "Faisalabad",
                                 "Multan", "Peshawar", "Gujranwala", "Quetta", "Sialkot", "Other"])

    if condition == "New":
        mileage = 0
        st.success("New car → Mileage set to **0 km**")
    else:
        mileage = st.number_input(
            "Mileage (km)",
            min_value=1, max_value=500000, value=30000, step=5000,
            help="Check the odometer on your dashboard. If unsure, enter approximate."
        )

# ==================== PREDICTION ====================
if st.button("Predict Price in Pakistan", type="primary", use_container_width=True):
    df = pd.DataFrame([{
        'brand': brand, 'model': model_input, 'year': year, 'mileage': mileage,
        'condition': condition, 'transmission': transmission, 'fuel': fuel, 'city': city,
        'age': 2025 - year,
        'log_mileage': np.log1p(mileage),
        'mileage_per_year': mileage / max((2025 - year), 1),
        'is_automatic': 1 if transmission == "Automatic" else 0,
        'is_hybrid_or_ev': 1 if fuel in ["Hybrid", "Electric"] else 0,
        'is_imported': 1 if condition == "Imported" else 0,
        'is_new_car': 1 if condition == "New" else 0,
        'city_premium': 1 if city in ["Lahore", "Islamabad", "Karachi"] else 0,
        'brand_tier': 3 if brand in ["Toyota","Honda"] 
                     else (2 if brand in ["Kia","Hyundai","MG","Mercedes","BMW","Audi"] else 1),
        'is_top_model': 1 if model_input.lower() in [
            "civic","city","corolla","yaris","sportage","tucson","fortuner","revo",
            "grande","prius","aqua","vezel","hr-v"
        ] else 0
    }])

    try:
        X = preprocessor.transform(df)
        pred = float(model.predict(X)[0])

        # DYNAMIC CONFIDENCE
        confidence = 92
        if condition == "New" and year != 2025: confidence -= 50
        if mileage == 0 and condition != "New": confidence -= 35
        if year < 2015: confidence -= 12
        if mileage > 200000: confidence -= 10
        if brand == "Other": confidence -= 25
        if brand in ["Mercedes", "BMW", "Audi"] and year < 2018: confidence -= 20
        confidence = max(55, min(98, confidence))

        # CRORE + LAKH DISPLAY (
        if pred >= 10_000_000:  # 1 crore or more
            crore = pred // 10_000_000
            remaining_lakh = (pred % 10_000_000) / 100_000
            if remaining_lakh >= 1:
                urdu_price = f"{crore:,} crore {remaining_lakh:.1f} لاکھ روپے"
            else:
                urdu_price = f"{crore:,} crore روپے"
        else:
            urdu_price = f"{pred/100_000:.1f} لاکھ روپے"

        # FINAL RESULT
        st.markdown("---")
        st.markdown(f"""
            <h1 style='text-align: center; color: #FF4B4B; margin-bottom: 10px;'>
                PKR {pred:,.0f}
            </h1>
            <h2 style='text-align: center; color: #2E8B57; margin-top: -10px;'>
                ≈ {urdu_price}
            </h2>
        """, unsafe_allow_html=True)

        # CONFIDENCE
        if confidence >= 90:
            st.success(f"Model is **{confidence}% confident** — Very reliable prediction")
            st.balloons()
        elif confidence >= 80:
            st.success(f"Model is **{confidence}% confident** — Good estimate")
            st.balloons()
        elif confidence >= 70:
            st.warning(f"Model is **{confidence}% confident** — Reasonable estimate")
        else:
            st.warning(f"Model is **{confidence}% confident** — This car is quite rare")

        st.caption("Prices are based on real 2024–2025 market listings • Last updated Dec 2025")

    except Exception as e:
        st.error("This combination is very rare — accurate prediction not possible.")
        st.warning("Model confidence: **Below 50%**")

# ==================== FOOTER ====================
st.markdown("---")

st.caption("Built by **Muhammad Haris Afridi** • 2025")
