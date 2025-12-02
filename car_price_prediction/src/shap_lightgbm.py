# src/shap_lightgbm.py
import pandas as pd
import joblib
import os
import shap
import matplotlib.pyplot as plt

# -------------------------------
# Paths
# -------------------------------
PROCESSED_DATA_PATH = r"D:\PYTHON\Project\car_price_prediction\data\processed\cleaned_pakistan_cars_5000_2025.csv"
MODEL_PATH          = r"D:\PYTHON\Project\car_price_prediction\models\model.pkl"
SHAP_SUMMARY_PATH   = r"D:\PYTHON\Project\car_price_prediction\models\shap_summary.png"

print("Loading model and data...")
full_pipeline = joblib.load(MODEL_PATH)
preprocessor = full_pipeline['preprocessor']
model = full_pipeline['model']
feature_names = full_pipeline['feature_names']

df = pd.read_csv(PROCESSED_DATA_PATH)
X = df.drop('price', axis=1)
y = df['price'].astype(float)

print("Transforming features...")
X_processed = preprocessor.transform(X).astype(float)

# Use a small background dataset for SHAP 
background = X_processed[:100]

print("Generating SHAP explanations for LightGBM...")
explainer = shap.TreeExplainer(model, data=background)
shap_values = explainer.shap_values(X_processed)

# Summary plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_processed, feature_names=feature_names, show=False)
plt.title("Top Features Affecting Car Price (SHAP)")
os.makedirs(os.path.dirname(SHAP_SUMMARY_PATH), exist_ok=True)
plt.savefig(SHAP_SUMMARY_PATH, dpi=200, bbox_inches='tight')
plt.close()

print(f"SHAP summary plot saved → {SHAP_SUMMARY_PATH}")
print("Done! You can now use this plot for analysis or your notebook.")
