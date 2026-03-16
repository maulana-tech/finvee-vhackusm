#!/usr/bin/env python

"""
AEGIS SME - RETRAIN WITH IEEE CIS
TIME FINVEE | VARSITY Hackathon 2026

This Script Retrains the AEGIS SME Model with IEEE CIS Dataset
1. Download the IEEE CIS Dataset
2. Preprocess the Dataset
3. Retrain the Model
4. Save the Model

"""

import os
import gc
import json
import pickle
import warnings
import numpy as np
import pandas as pd 
import lightgbm as lgb
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

# Setup and Configuration
DATA_PATH = "./data"
MODEL_PATH = "./models"
SUBSAMPLE_FRAC = 0.25

# Download the IEEE CIS Dataset
print("=" * 60)
print("Downloading the IEEE CIS Dataset")
print("=" * 60)

if not os.path.exists(os.path.join(DATA_PATH, 'train_transaction.csv')):
    print("\n[CMD] Running Kaggle CLI to download data...")
    # Check for kaggle.json
    if not os.path.exists('/home/ubuntu/.kaggle/kaggle.json'):
        print("\n[ERROR] `kaggle.json` not found in `/home/ubuntu/.kaggle/`.")
        print("Please follow the prerequisite steps in the script's docstring.")
        exit()

    # Install kaggle if not present
    os.system("sudo pip3 install kaggle --quiet")

    # Download and unzip
    command = f"kaggle competitions download -c ieee-fraud-detection -p {DATA_PATH} --force && \
                unzip -o {DATA_PATH}ieee-fraud-detection.zip -d {DATA_PATH} && \
                rm {DATA_PATH}ieee-fraud-detection.zip"
    exit_code = os.system(command)
    if exit_code != 0:
        print("\n[ERROR] Failed to download data from Kaggle. Please check your API key and permissions.")
        exit()
    print("\n[OK] Dataset downloaded and unzipped successfully.")
else:
    print("\n[OK] Dataset already exists. Skipping download.")


# LOAD AND PREPROCESS THE DATASET

print("\n" + "=" * 60)
print("  STEP 2: Loading and Preprocessing Data")
print("=" * 60)

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f'[MEM] Memory usage reduced to {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    return df

print("\n[LOAD] Loading train_transaction and train_identity...")
train_txn = pd.read_csv(os.path.join(DATA_PATH, 'train_transaction.csv'))
train_id = pd.read_csv(os.path.join(DATA_PATH, 'train_identity.csv'))

print("\n[MERGE] Merging datasets...")
train_df = pd.merge(train_txn, train_id, on='TransactionID', how='left')

del train_txn, train_id
gc.collect()

print(f"\n[SUBSAMPLE] Using {SUBSAMPLE_FRAC:.0%} of the data for training.")
train_df = train_df.sample(frac=SUBSAMPLE_FRAC, random_state=42).reset_index(drop=True)

train_df = reduce_mem_usage(train_df)
print(f"\n[DATA] Loaded {len(train_df)} transactions.")
print(f"  Fraud rate: {train_df['isFraud'].mean():.2%}")

# Feature Engineering & Selection
print("\n[FEAT] Performing feature engineering and selection...")

# Based on public notebooks, these are generally useful features
# We select a small subset to keep training fast for a hackathon
USEFUL_FEATURES = [
    'TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
    'addr1', 'addr2', 'dist1', 'dist2', 'P_emaildomain', 'R_emaildomain',
    'C1', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14',
    'D1', 'D2', 'D3', 'D4', 'D5', 'D10', 'D15',
    'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
    'id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_11',
    'id_12', 'id_13', 'id_15', 'id_17', 'id_19', 'id_20', 'id_30', 'id_31', 'id_38',
    'DeviceType', 'DeviceInfo'
]

# Select only useful features that exist in the dataframe
features_to_use = [f for f in USEFUL_FEATURES if f in train_df.columns]
print(f"  Selected {len(features_to_use)} potentially useful features.")

X = train_df[features_to_use]
y = train_df['isFraud']

del train_df
gc.collect()

# Label Encoding for categorical features
categorical_features = X.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_features:
    X[col] = X[col].astype(str)
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Fill NaNs (simple strategy for speed)
X = X.fillna(-999)

print("  Feature processing complete.")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"[DATA] Train: {len(X_train)} | Test: {len(X_test)}")

# ─────────────────────────────────────────────
# 3. RETRAIN MODELS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 3: Retraining Models")
print("=" * 60)

# --- LightGBM Classifier ---
print("\n[MODEL 1] Training LightGBM Classifier on IEEE-CIS data...")

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'n_estimators': 1000,  # Increased for larger dataset
    'learning_rate': 0.02,
    'num_leaves': 491,
    'max_depth': 16,
    'seed': 42,
    'n_jobs': -1,
    'verbose': -1,
    'colsample_bytree': 0.7,
    'subsample': 0.7,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
}

lgb_model = lgb.LGBMClassifier(**lgb_params)
lgb_model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_test_scaled, y_test)],
    callbacks=[lgb.early_stopping(100, verbose=True), lgb.log_evaluation(100)]
)

lgb_proba = lgb_model.predict_proba(X_test_scaled)[:, 1]
lgb_pred = (lgb_proba > 0.5).astype(int)
lgb_auc = roc_auc_score(y_test, lgb_proba)
lgb_ap = average_precision_score(y_test, lgb_proba)

print(f"\n  LightGBM AUC-ROC:       {lgb_auc:.4f}")
print(f"  LightGBM Avg Precision: {lgb_ap:.4f}")
print(classification_report(y_test, lgb_pred, target_names=['Normal', 'Fraud']))

# --- Autoencoder ---
print("\n[MODEL 2] Training Autoencoder on IEEE-CIS data...")

# Train ONLY on normal transactions
X_normal = X_train_scaled[y_train == 0]

input_dim = X_normal.shape[1]

autoencoder = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(input_dim,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(input_dim, activation='linear')
])

autoencoder.compile(optimizer='adam', loss='mse')

history = autoencoder.fit(
    X_normal, X_normal,
    epochs=100,
    batch_size=512,
    validation_split=0.1,
    verbose=0,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    ]
)
print(f"  Training complete | Final loss: {history.history['loss'][-1]:.6f}")

# Compute reconstruction error threshold
X_normal_recon = autoencoder.predict(X_normal, verbose=0, batch_size=2048)
normal_errors = np.mean(np.square(X_normal - X_normal_recon), axis=1)
ae_threshold = np.percentile(normal_errors, 98) # Use 98th percentile for more complex data
print(f"  Anomaly threshold (98th pct): {ae_threshold:.6f}")

# Evaluate Autoencoder
X_test_recon = autoencoder.predict(X_test_scaled, verbose=0, batch_size=2048)
ae_errors = np.mean(np.square(X_test_scaled - X_test_recon), axis=1)
ae_auc = roc_auc_score(y_test, ae_errors)

print(f"  Autoencoder AUC-ROC (error as score): {ae_auc:.4f}")

# ─────────────────────────────────────────────
# 4. SAVE NEW ARTIFACTS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"  STEP 4: Saving New Artifacts to {MODEL_PATH}")
print("=" * 60)

with open(os.path.join(MODEL_PATH, "lgb_model_ieee.pkl"), "wb") as f:
    pickle.dump(lgb_model, f)
autoencoder.save(os.path.join(MODEL_PATH, "autoencoder_ieee.keras"))
with open(os.path.join(MODEL_PATH, "scaler_ieee.pkl"), "wb") as f:
    pickle.dump(scaler, f)
with open(os.path.join(MODEL_PATH, "label_encoders_ieee.pkl"), "wb") as f:
    pickle.dump(label_encoders, f)

metadata = {
    "dataset": "IEEE-CIS Fraud Detection",
    "subsample_frac": SUBSAMPLE_FRAC,
    "features": features_to_use,
    "ae_threshold": float(ae_threshold),
    "lgb_auc": float(lgb_auc),
    "ae_auc": float(ae_auc),
}
with open(os.path.join(MODEL_PATH, "metadata_ieee.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print("  [OK] lgb_model_ieee.pkl")
print("  [OK] autoencoder_ieee.keras")
print("  [OK] scaler_ieee.pkl")
print("  [OK] label_encoders_ieee.pkl")
print("  [OK] metadata_ieee.json")

print("\n" + "=" * 60)
print("  Retraining Complete!")
print(f"  New models saved in: {MODEL_PATH}")
print(f"  LightGBM AUC:  {lgb_auc:.4f}")
print(f"  Autoencoder AUC: {ae_auc:.4f}")
print("=" * 60)
