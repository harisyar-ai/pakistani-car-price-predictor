# src/data_utils.py  
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import os

# -------------------------------
# Logging setup
# -------------------------------
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# -------------------------------
# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # src/ → project root

RAW_DATA_PATH = Path(r"D:\PYTHON\Project\car_price_prediction\data\raw\pakistan_cars_5000_PERFECT_2025.csv")
PROCESSED_DATA_PATH = Path(r"D:\PYTHON\Project\car_price_prediction\data\processed\cleaned_pakistan_cars_5000_2025.csv")

def load_and_clean_data() -> pd.DataFrame:
    logging.info(f"Checking raw file: {RAW_DATA_PATH}")
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"RAW FILE MISSING! Expected: {RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH)
    logging.info(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    # Rename for consistency
    df = df.rename(columns={
        'year_of_manufacture': 'year',
        'mileage_km': 'mileage',
        'price_pkr': 'price',
        'fuel_type': 'fuel',
        'registration_city': 'city'
    })

    # Basic cleaning
    df = df.dropna(subset=['price'])
    df['price'] = df['price'].astype(float)

    # === FEATURE ENGINEERING ===
    current_year = 2025
    df['age'] = current_year - df['year']
    df['mileage_per_year'] = df['mileage'] / (df['age'] + 1)
    df['is_new_car'] = ((df['mileage'] < 1000) & (df['condition'] == 'New')).astype(int)
    df['is_imported'] = (df['condition'] == 'Imported').astype(int)
    df['is_automatic'] = (df['transmission'] == 'Automatic').astype(int)
    df['is_hybrid_or_ev'] = df['fuel'].isin(['Hybrid', 'Electric']).astype(int)

    premium_brands = ['Toyota', 'Honda', 'Mercedes', 'BMW', 'Audi']
    mid_brands = ['Kia', 'Hyundai', 'MG', 'Haval']
    df['brand_tier'] = df['brand'].apply(
        lambda x: 3 if x in premium_brands else (2 if x in mid_brands else 1)
    )

    df['city_premium'] = df['city'].isin(['Lahore', 'Karachi']).astype(int)
    df['log_mileage'] = np.log1p(df['mileage'])
    top_models = ['Corolla', 'City', 'Civic', 'Yaris', 'Sportage', 'Fortuner', 'Prius']
    df['is_top_model'] = df['model'].isin(top_models).astype(int)

    # Final features
    feature_cols = [
        'brand', 'model', 'year', 'mileage', 'condition', 'transmission',
        'fuel', 'city', 'age', 'mileage_per_year', 'is_new_car',
        'is_imported', 'is_automatic', 'is_hybrid_or_ev', 'brand_tier',
        'city_premium', 'log_mileage', 'is_top_model'
    ]

    df_final = df[feature_cols + ['price']].copy()
    df_final = df_final[df_final['price'] > 300000]


    os.makedirs(PROCESSED_DATA_PATH.parent, exist_ok=True)
    df_final.to_csv(PROCESSED_DATA_PATH, index=False)
    
    if PROCESSED_DATA_PATH.exists():
        logging.info(f"SUCCESS! File saved at: {PROCESSED_DATA_PATH}")
        logging.info(f"Size: {PROCESSED_DATA_PATH.stat().st_size / 1024:.1f} KB")
    else:
        raise FileNotFoundError("SAVE FAILED — check permissions or disk space")

    return df_final

if __name__ == "__main__":
    df_clean = load_and_clean_data()
    print(df_clean.head())
    print(f"\nFINAL SHAPE: {df_clean.shape}")
