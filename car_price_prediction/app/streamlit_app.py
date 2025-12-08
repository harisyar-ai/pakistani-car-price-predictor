# app/streamlit_app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import streamlit.components.v1 as components

# --------------------- PATHS ---------------------
BASE_DIR = Path(__file__).parent.parent
PROFILE_PATH = BASE_DIR / "profile.png"

# --------------------- PARTICLES + DESIGN ---------------------
components.html("""
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap">
<style>
    :root {--bg: #0e1117; --accent: #ff4b4b; --accent2: #ff7b00;}
    html, body {background: var(--bg) !important; font-family: 'Inter', sans-serif; color: #e0e0e0;}
    .header h1 {color: var(--accent); text-align: center; font-size: 48px; margin: 0;}
    .header p {color: #999; text-align: center; font-size: 18px;}
    .card {background: rgba(255,255,255,0.05); border-radius: 16px; padding: 28px; border: 1px solid rgba(255,75,75,0.15); backdrop-filter: blur(12px);}
    .profile-img {width: 130px; height: 130px; border-radius: 50%; object-fit: cover; border: 4px solid var(--accent); box-shadow: 0 10px 40px rgba(255,75,75,0.3);}
    .social-link {color: #ff6b6b;; text-decoration: none; font-weight: 600; position: relative;}
    .social-link::after {content: ""; position: absolute; width: 100%; height: 3px; bottom: -6px; left: 0; background: linear-gradient(90deg, var(--accent), var(--accent2)); transform: scaleX(0); transition: transform 0.4s ease;}
    .social-link:hover::after {transform: scaleX(1);}
    #particles-js {position: fixed; width: 100%; height: 100%; top: 0; left: 0; z-index: 0; opacity: 0.65;}
    .main > div {position: relative; z-index: 1;}
</style>
<div id="particles-js"></div>
<script src="https://cdn.jsdelivr.net/npm/tsparticles@2.12.0/tsparticles.bundle.min.js"></script>
<script>
tsParticles.load("particles-js", {
  particles: {number: {value: 80}, color: {value: ["#ff4b4b", "#ff7b00"]}, size: {value: {min: 1, max: 5}},
    move: {enable: true, speed: 0.7}, links: {enable: true, distance: 140, color: "#ff4b4b", opacity: 0.1}},
  interactivity: {events: {onHover: {enable: true, mode: "grab"}}}
});
</script>
""", height=0)

# --------------------- SIDEBAR ---------------------
with st.sidebar:
    st.markdown("<h2 style='color:#ff4b4b; text-align:center;'>Pakistani Car Price Predictor</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#999;'>5,000+ verified listings • 2025</p>", unsafe_allow_html=True)
    st.markdown("---")
    if PROFILE_PATH.exists():
        st.image(str(PROFILE_PATH), use_column_width=True, caption="Muhammad Haris Afridi")
    else:
        st.image("https://i.imgur.com/3j8BPdc.png", use_column_width=True)
    st.markdown("<p style='text-align:center;'><strong>Muhammad Haris Afridi</strong><br><small>AI Engineer & ML Developer</small></p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Connect with me")
    st.markdown("<a class='social-link' href='https://github.com/harisyar-ai' target='_blank'>GitHub → github.com/harisyar-ai</a><br>", unsafe_allow_html=True)
    st.markdown("<a class='social-link' href='https://www.linkedin.com/in/muhammad-haris-afridi-051395379' target='_blank'>LinkedIn → Muhammad Haris Afridi</a><br>", unsafe_allow_html=True)
    st.markdown("<span style='color:#ff6b6b;'>Email →</span> mharisyar.ai@gmail.com", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("© 2025 • Built in Pakistan")

# ==================== LOAD MODEL ====================
@st.cache_resource
def load_model():
    pipeline = joblib.load(BASE_DIR / "models" / "model.pkl")
    model = pipeline['model']
    preprocessor = pipeline['preprocessor']
    return {'pipeline': pipeline, 'model': model, 'preprocessor': preprocessor}

loaded = load_model()
pipeline = loaded['pipeline']
model = loaded['model']
preprocessor = loaded['preprocessor']

# ==================== HEADER ====================
st.markdown("""
    <div class="header">
        <h1>Pakistani Car Price Predictor</h1>
        <p>Based on 5,000+ verified listings • Real-time 2025 prices</p>
    </div>
    <hr style='border: 2px solid #ff4b4b; border-radius: 5px; margin: 40px 0;'>
""", unsafe_allow_html=True)

# ==================== FULL DICTIONARIES (100% YOURS) ====================
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

model_specs = {
    "Prius": {"fuel": ["Hybrid"], "transmission": ["Automatic"]},
    "Aqua": {"fuel": ["Hybrid"], "transmission": ["Automatic"]},
    "Vezel": {"fuel": ["Hybrid", "Petrol"], "transmission": ["Automatic"]},
    "HR-V": {"fuel": ["Petrol"], "transmission": ["Automatic", "Manual"]},
    "Civic": {"fuel": ["Petrol"], "transmission": ["Automatic", "Manual"]},
    "City": {"fuel": ["Petrol"], "transmission": ["Automatic", "Manual"]},
    "Corolla": {"fuel": ["Petrol"], "transmission": ["Automatic", "Manual"]},
    "Yaris": {"fuel": ["Petrol"], "transmission": ["Automatic", "Manual"]},
    "Alto": {"fuel": ["Petrol"], "transmission": ["Manual", "Automatic"]},
    "Cultus": {"fuel": ["Petrol", "CNG"], "transmission": ["Manual"]},
    "Wagon R": {"fuel": ["Petrol"], "transmission": ["Manual", "Automatic"]},
    "Mehran": {"fuel": ["Petrol", "CNG"], "transmission": ["Manual"]},
    "Sportage": {"fuel": ["Petrol", "Diesel"], "transmission": ["Automatic"]},
    "Tucson": {"fuel": ["Petrol", "Diesel"], "transmission": ["Automatic"]},
    "Fortuner": {"fuel": ["Diesel"], "transmission": ["Automatic", "Manual"]},
    "Revo": {"fuel": ["Diesel"], "transmission": ["Automatic", "Manual"]},
    "Hilux": {"fuel": ["Diesel"], "transmission": ["Automatic", "Manual"]},
    "Picanto": {"fuel": ["Petrol"], "transmission": ["Manual", "Automatic"]},
    "Stonic": {"fuel": ["Petrol"], "transmission": ["Automatic"]},
    "Elantra": {"fuel": ["Petrol"], "transmission": ["Automatic"]},
    "Sonata": {"fuel": ["Petrol"], "transmission": ["Automatic"]},
    "C-Class": {"fuel": ["Petrol", "Diesel"], "transmission": ["Automatic"]},
    "E-Class": {"fuel": ["Petrol", "Diesel"], "transmission": ["Automatic"]},
    "A3": {"fuel": ["Petrol", "Hybrid"], "transmission": ["Automatic"]},
    "Q5": {"fuel": ["Petrol", "Hybrid"], "transmission": ["Automatic"]},
    "H6": {"fuel": ["Petrol"], "transmission": ["Automatic"]},
    "Jolion": {"fuel": ["Petrol", "Hybrid"], "transmission": ["Automatic"]},
    "HS": {"fuel": ["Petrol"], "transmission": ["Automatic"]},
    "ZS": {"fuel": ["Petrol"], "transmission": ["Manual", "Automatic"]},
}

def get_fuel_options(model_name):
    return model_specs.get(model_name, {}).get("fuel", ["Petrol", "Hybrid", "Diesel", "CNG", "Electric"])

def get_transmission_options(model_name):
    return model_specs.get(model_name, {}).get("transmission", ["Automatic", "Manual"])

# ==================== INPUT FORM ====================
st.markdown("<div class='card'>", unsafe_allow_html=True)
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
        mileage = st.number_input("Mileage (km)", min_value=1, max_value=500000, value=30000, step=5000,
                                  help="Check the odometer on your dashboard. If unsure, enter approximate.")

st.markdown("</div>", unsafe_allow_html=True)

# ==================== PREDICTION (100% YOUR ORIGINAL) ====================
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

        # Your exact confidence logic
        confidence = 92
        if condition == "New" and year != 2025: confidence -= 50
        if mileage == 0 and condition != "New": confidence -= 35
        if year < 2015: confidence -= 12
        if mileage > 200000: confidence -= 10
        if brand == "Other": confidence -= 25
        if brand in ["Mercedes", "BMW", "Audi"] and year < 2018: confidence -= 20
        confidence = max(55, min(98, confidence))

        # Urdu price display
        if pred >= 10_000_000:
            crore = pred // 10_000_000
            remaining_lakh = (pred % 10_000_000) / 100_000
            urdu_price = f"{crore:,} کروڑ {remaining_lakh:.1f} لاکھ روپے" if remaining_lakh >= 1 else f"{crore:,} کروڑ روپے"
        else:
            urdu_price = f"{pred/100_000:.1f} لاکھ روپے"

        st.markdown("---")
        st.markdown(f"""
            <h1 style='text-align: center; color: #FF4B4B; margin-bottom: 10px;'>
                PKR {pred:,.0f}
            </h1>
            <h2 style='text-align: center; color: #2E8B57; margin-top: -10px;'>
                ≈ {urdu_price}
            </h2>
        """, unsafe_allow_html=True)

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

    except Exception as e:
        st.error("This combination is very rare — accurate prediction not possible.")
        st.warning("Model confidence: **Below 50%**")

# ==================== FOOTER ====================
st.markdown("---")
st.caption("Prices based on real 2024–2025 market data • Last updated December 2025")
st.caption("© 2025 Muhammad Haris Afridi • Built with passion in Pakistan")
