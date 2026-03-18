"""
AEGIS SME — ML Core Engine
Team Finvee | Varsity Hackathon 2026

Two-model ensemble:
1. LightGBM Classifier  → supervised fraud classification (risk score 0-1)
2. Autoencoder          → unsupervised behavioral anomaly detection
"""

import numpy as np
import pandas as pd
import pickle
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, roc_auc_score,
                              precision_recall_curve, average_precision_score,
                              confusion_matrix)
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_FILE = os.path.join(BASE_DIR, "data", "transactions.csv")

os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD & FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("=" * 60)
print("  AEGIS SME — ML Core Engine Training")
print("  Team Finvee | Varsity Hackathon 2026")
print("=" * 60)

df = pd.read_csv(DATA_FILE)
print(f"\n[DATA] Loaded {len(df)} transactions | Fraud rate: {df['is_fraud'].mean():.2%}")

# Encode categoricals
le_merchant_type = LabelEncoder()
le_location = LabelEncoder()
df['merchant_type_enc'] = le_merchant_type.fit_transform(df['merchant_type'])
df['location_enc'] = le_location.fit_transform(df['location'])

# Feature set
FEATURES = [
    'hour', 'day_of_week', 'amount', 'is_new_device',
    'transaction_count_1h', 'transaction_count_24h',
    'amount_vs_avg_ratio', 'location_mismatch',
    'merchant_type_enc', 'location_enc'
]

X = df[FEATURES].values
y = df['is_fraud'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"[DATA] Train: {len(X_train)} | Test: {len(X_test)}")

# ─────────────────────────────────────────────
# 2. LIGHTGBM CLASSIFIER
# ─────────────────────────────────────────────
print("\n[MODEL 1] Training LightGBM Classifier...")

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
print(f"  After SMOTE: {sum(y_train_resampled==0)} normal | {sum(y_train_resampled==1)} fraud")

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'scale_pos_weight': 1,
    'verbose': -1,
    'random_state': 42,
    'n_estimators': 300,
    'min_child_samples': 5,
}

lgb_model = lgb.LGBMClassifier(**lgb_params)
lgb_model.fit(
    X_train_resampled, y_train_resampled,
    eval_set=[(X_test_scaled, y_test)],
    callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(False)]
)

# Evaluate LightGBM
lgb_proba = lgb_model.predict_proba(X_test_scaled)[:, 1]
lgb_pred = (lgb_proba > 0.5).astype(int)
lgb_auc = roc_auc_score(y_test, lgb_proba)
lgb_ap = average_precision_score(y_test, lgb_proba)

print(f"  AUC-ROC:          {lgb_auc:.4f}")
print(f"  Avg Precision:    {lgb_ap:.4f}")
print(f"\n  Classification Report:")
print(classification_report(y_test, lgb_pred, target_names=['Normal', 'Fraud']))

# Feature importance
fi = pd.DataFrame({
    'feature': FEATURES,
    'importance': lgb_model.feature_importances_
}).sort_values('importance', ascending=False)
print(f"\n  Top Features:\n{fi.to_string(index=False)}")

# ─────────────────────────────────────────────
# 3. AUTOENCODER (Behavioral Anomaly Detection)
# ─────────────────────────────────────────────
print("\n[MODEL 2] Training Autoencoder (Behavioral Anomaly Detector)...")

# Train ONLY on normal transactions
X_normal = X_train_scaled[y_train == 0]
print(f"  Training on {len(X_normal)} normal transactions only")

input_dim = X_normal.shape[1]

# Build Autoencoder
inputs = keras.Input(shape=(input_dim,))
# Encoder
x = layers.Dense(16, activation='relu')(inputs)
x = layers.Dropout(0.1)(x)
x = layers.Dense(8, activation='relu')(x)
encoded = layers.Dense(4, activation='relu', name='bottleneck')(x)
# Decoder
x = layers.Dense(8, activation='relu')(encoded)
x = layers.Dropout(0.1)(x)
x = layers.Dense(16, activation='relu')(x)
decoded = layers.Dense(input_dim, activation='linear')(x)

autoencoder = keras.Model(inputs, decoded, name='AegisAutoencoder')
autoencoder.compile(optimizer='adam', loss='mse')

history = autoencoder.fit(
    X_normal, X_normal,
    epochs=50,
    batch_size=64,
    validation_split=0.1,
    verbose=0,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=0)
    ]
)
print(f"  Training complete | Final loss: {history.history['loss'][-1]:.6f}")

# Compute reconstruction error threshold
X_normal_recon = autoencoder.predict(X_normal, verbose=0)
normal_errors = np.mean(np.square(X_normal - X_normal_recon), axis=1)
# Set threshold at 95th percentile of normal errors
ae_threshold = np.percentile(normal_errors, 95)
print(f"  Anomaly threshold (95th pct): {ae_threshold:.6f}")

# Evaluate Autoencoder on test set
X_test_recon = autoencoder.predict(X_test_scaled, verbose=0)
ae_errors = np.mean(np.square(X_test_scaled - X_test_recon), axis=1)
ae_pred = (ae_errors > ae_threshold).astype(int)
ae_auc = roc_auc_score(y_test, ae_errors)

print(f"  AUC-ROC (error as score): {ae_auc:.4f}")
print(f"\n  Classification Report:")
print(classification_report(y_test, ae_pred, target_names=['Normal', 'Fraud']))

# ─────────────────────────────────────────────
# 4. ENSEMBLE DECISION ENGINE
# ─────────────────────────────────────────────
print("\n[ENSEMBLE] Testing Ensemble Decision Engine...")

def ensemble_score(lgb_score, ae_error, ae_threshold, lgb_weight=0.65, ae_weight=0.35):
    """Combine LightGBM risk score and Autoencoder anomaly score."""
    ae_score = min(ae_error / (ae_threshold * 3), 1.0)  # Normalize AE error to [0,1]
    combined = lgb_weight * lgb_score + ae_weight * ae_score
    return float(combined)

def get_decision(score):
    """Convert ensemble score to action decision."""
    if score >= 0.75:
        return "BLOCK", "HIGH"
    elif score >= 0.45:
        return "STEP_UP_AUTH", "MEDIUM"
    else:
        return "APPROVE", "LOW"

# Test ensemble
ensemble_scores = [
    ensemble_score(lgb_proba[i], ae_errors[i], ae_threshold)
    for i in range(len(lgb_proba))
]
ensemble_pred = [1 if s >= 0.45 else 0 for s in ensemble_scores]
ensemble_auc = roc_auc_score(y_test, ensemble_scores)

print(f"  Ensemble AUC-ROC: {ensemble_auc:.4f}")
print(f"\n  Ensemble Classification Report:")
print(classification_report(y_test, ensemble_pred, target_names=['Normal', 'Fraud']))

# ─────────────────────────────────────────────
# 5. SAVE ALL ARTIFACTS
# ─────────────────────────────────────────────
print("\n[SAVE] Saving model artifacts...")

# Save LightGBM
with open(os.path.join(MODEL_DIR, "lgb_model.pkl"), "wb") as f:
    pickle.dump(lgb_model, f)

# Save Autoencoder
autoencoder.save(os.path.join(MODEL_DIR, "autoencoder.keras"))

# Save Scaler & Encoders
with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
with open(os.path.join(MODEL_DIR, "label_encoders.pkl"), "wb") as f:
    pickle.dump({'merchant_type': le_merchant_type, 'location': le_location}, f)

# Save metadata
metadata = {
    "features": FEATURES,
    "ae_threshold": float(ae_threshold),
    "lgb_auc": float(lgb_auc),
    "ae_auc": float(ae_auc),
    "ensemble_auc": float(ensemble_auc),
    "decision_thresholds": {"block": 0.75, "step_up": 0.45, "approve": 0.0},
    "lgb_weight": 0.65,
    "ae_weight": 0.35,
    "merchant_types": list(le_merchant_type.classes_),
    "locations": list(le_location.classes_)
}
with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print("  [OK] lgb_model.pkl")
print("  [OK] autoencoder.keras")
print("  [OK] scaler.pkl")
print("  [OK] label_encoders.pkl")
print("  [OK] metadata.json")

print("\n" + "=" * 60)
print("  ML Core Engine Training Complete!")
print(f"  LightGBM AUC:  {lgb_auc:.4f}")
print(f"  Autoencoder AUC: {ae_auc:.4f}")
print(f"  Ensemble AUC:  {ensemble_auc:.4f}")
print("=" * 60)
