# src/train.py  ← 8 MODELS, FULL COMPARISON, BEST ONE SAVED
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# === 8 MODELS ===
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
import xgboost as xgb
import lightgbm as lgb

# -------------------------------
# Paths
# -------------------------------
PROCESSED_DATA_PATH = r"D:\PYTHON\Project\car_price_prediction\data\processed\cleaned_pakistan_cars_5000_2025.csv"
PREPROCESSOR_PATH   = r"D:\PYTHON\Project\car_price_prediction\models\preprocessor.pkl"
FEATURE_NAMES_PATH  = r"D:\PYTHON\Project\car_price_prediction\models\feature_names.pkl"
FINAL_MODEL_PATH    = r"D:\PYTHON\Project\car_price_prediction\models\model.pkl"

print("Loading data...")
df = pd.read_csv(PROCESSED_DATA_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)
feature_names = joblib.load(FEATURE_NAMES_PATH)

X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_p = preprocessor.transform(X_train)
X_test_p  = preprocessor.transform(X_test)

# === THE 8 MODELS YOU WANTED ===
models = {
    "1. Linear Regression"     : LinearRegression(),
    "2. Ridge Regression"      : Ridge(alpha=1.0),
    "3. Decision Tree          " : DecisionTreeRegressor(max_depth=12, random_state=42),
    "4. Extra Trees            " : ExtraTreesRegressor(n_estimators=600, max_depth=15, n_jobs=-1, random_state=42),
    "5. Random Forest          " : RandomForestRegressor(n_estimators=600, max_depth=15, n_jobs=-1, random_state=42),
    "6. Gradient Boosting      " : GradientBoostingRegressor(n_estimators=800, learning_rate=0.05, max_depth=7, random_state=42),
    "7. XGBoost                " : xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=8, subsample=0.9, colsample_bytree=0.8, random_state=42, n_jobs=-1, tree_method='hist'),
    "8. LightGBM               " : lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, max_depth=8, num_leaves=80, random_state=42, n_jobs=-1, verbose=-1),
}

print("\nTraining 8 models — please wait 30–60 seconds...\n" + "—" * 70)

results = []
best_r2 = -1
best_model = None
best_name = ""

for name, model in models.items():
    print(f"{name} → training...", end="")
    model.fit(X_train_p, y_train)
    pred = model.predict(X_test_p)
    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    print(f" R² = {r2:.4f} | MAE = ±{mae:,.0f} PKR")
    
    results.append((name, r2, mae, model))
    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_name = name

# === FINAL RANKING ===
print("\n" + "="*80)
print("FINAL RANKING — 8 MODELS COMPARED")
print("="*80)
results.sort(key=lambda x: x[1], reverse=True)
for i, (name, r2, mae, _) in enumerate(results, 1):
    trophy = "WINNER" if i == 1 else "      "
    print(f"{i}. {name} → R² = {r2:.4f} | MAE = ±{mae:,.0f} PKR  {trophy}")

# === SAVE THE BEST ONE ===
full_pipeline = {
    'preprocessor': preprocessor,
    'model': best_model,
    'feature_names': feature_names,
    'model_name': best_name.split('.')[1].strip()
}

joblib.dump(full_pipeline, FINAL_MODEL_PATH)
print(f"\nBEST MODEL SAVED → {best_name} with R² = {best_r2:.4f}")
print(f"File: {FINAL_MODEL_PATH}")
