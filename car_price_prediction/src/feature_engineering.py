# src/train.py 
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt

# -------------------------------
# Paths
# -------------------------------
PROCESSED_DATA_PATH = r"D:\PYTHON\Project\car_price_prediction\data\processed\cleaned_pakistan_cars_5000_2025.csv"
PREPROCESSOR_PATH   = r"D:\PYTHON\Project\car_price_prediction\models\preprocessor.pkl"
FEATURE_NAMES_PATH  = r"D:\PYTHON\Project\car_price_prediction\models\feature_names.pkl"
FINAL_MODEL_PATH    = r"D:\PYTHON\Project\car_price_prediction\models\model.pkl"
SHAP_PLOT_PATH      = r"D:\PYTHON\Project\car_price_prediction\models\shap_summary.png"

print("Loading data & preprocessor...")
df = pd.read_csv(PROCESSED_DATA_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)
feature_names = joblib.load(FEATURE_NAMES_PATH)

X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_proc = preprocessor.transform(X_train)
X_test_proc = preprocessor.transform(X_test)

# -------------------------------
# Models
# -------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Decision Tree": DecisionTreeRegressor(max_depth=12, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=500, max_depth=15, n_jobs=-1, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=600, learning_rate=0.05, max_depth=6, random_state=42),
    "XGBoost": xgb.XGBRegressor(n_estimators=800, learning_rate=0.05, max_depth=7, subsample=0.9,
                                colsample_bytree=0.8, random_state=42, n_jobs=-1, tree_method='hist'),  # ← fixes SHAP
    "LightGBM": lgb.LGBMRegressor(n_estimators=800, learning_rate=0.05, max_depth=8, num_leaves=64,
                                  random_state=42, n_jobs=-1, verbose=-1),
}

results = []
best_model = None
best_r2 = -1
best_name = ""

print("\nTraining 7 models...\n" + "="*60)
for name, model in models.items():
    print(f"Training {name}...", end="")
    model.fit(X_train_proc, y_train)
    pred = model.predict(X_test_proc)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    print(f" → R² = {r2:.4f} | MAE = ±PKR {mae:,.0f}")
    results.append((name, mae, r2, model))
    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_name = name

# -------------------------------
# Final Results
# -------------------------------
print("\n" + "="*60)
print("RANKING (Best to Worst)")
print("="*60)
results.sort(key=lambda x: x[2], reverse=True)
for i, (name, mae, r2, _) in enumerate(results, 1):
    star = " ← WINNER!" if name == best_name else ""
    print(f"{i}. {name:18} → R² = {r2:.4f} | MAE = ±PKR {mae:,.0f}{star}")

# -------------------------------
# Save best model
# -------------------------------
full_pipeline = {
    'preprocessor': preprocessor,
    'model': best_model,
    'feature_names': feature_names,
    'model_name': best_name
}
joblib.dump(full_pipeline, FINAL_MODEL_PATH)
print(f"\nBEST MODEL SAVED: {best_name} (R² = {best_r2:.4f}) → {FINAL_MODEL_PATH}")

# -------------------------------
# SHAP 
# -------------------------------
print(f"\nGenerating SHAP plot for {best_name}...")
try:
    if "XGBoost" in best_name:
        explainer = shap.TreeExplainer(best_model, feature_perturbation="interventional")
    elif "LightGBM" in best_name:
        explainer = shap.TreeExplainer(best_model)
    elif "Gradient Boosting" in best_name or "Random Forest" in best_name:
        explainer = shap.TreeExplainer(best_model)
    else:
        explainer = shap.KernelExplainer(best_model.predict, X_train_proc[:100])

    shap_values = explainer.shap_values(X_test_proc[:200])  # sample for speed

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test_proc[:200], feature_names=feature_names, show=False)
    plt.title(f"SHAP Summary - {best_name} (R² = {best_r2:.4f})")
    plt.tight_layout()
    plt.savefig(SHAP_PLOT_PATH, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"SHAP plot saved → {SHAP_PLOT_PATH}")
except Exception as e:
    print(f"SHAP failed (non-critical): {e}")

print("\nTRAINING 100% COMPLETE!")
