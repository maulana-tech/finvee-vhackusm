#!/usr/bin/env python3
"""
AEGIS SME — Retraining Script for IEEE-CIS Dataset
Team Finvee | Varsity Hackathon 2026

All training outputs, metrics, and visualizations are automatically
saved as high-quality images in the /output folder for jury review.

**PREREQUISITES:**
1. You MUST have a Kaggle account.
2. You MUST create a Kaggle API token (`kaggle.json`).
3. You MUST place the `kaggle.json` file in `~/.kaggle/kaggle.json`.

   Steps:
   a. mkdir -p ~/.kaggle
   b. Paste your API key: {"username":"YOUR_USER","key":"YOUR_KEY"}
      into ~/.kaggle/kaggle.json
   c. chmod 600 ~/.kaggle/kaggle.json
   d. Accept competition rules at: https://www.kaggle.com/c/ieee-fraud-detection
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve, f1_score,
    precision_score, recall_score
)

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# ─────────────────────────────────────────────
# 0. SETUP & CONFIGURATION
# ─────────────────────────────────────────────
DATA_PATH   = os.path.join(os.path.dirname(__file__), "data", "ieee-cis")
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "models")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "output")
SUBSAMPLE_FRAC = 0.25   # 0.25 = 25% of data (fast). Set to 1.0 for full dataset.

for p in [DATA_PATH, MODEL_PATH, OUTPUT_PATH]:
    os.makedirs(p, exist_ok=True)

# ── Plotting style ──────────────────────────
FINVEE_DARK   = "#0D1117"
FINVEE_CARD   = "#161B22"
FINVEE_ACCENT = "#58A6FF"
FINVEE_GREEN  = "#3FB950"
FINVEE_RED    = "#F85149"
FINVEE_YELLOW = "#D29922"
FINVEE_TEXT   = "#C9D1D9"
FINVEE_MUTED  = "#8B949E"

plt.rcParams.update({
    'figure.facecolor':  FINVEE_DARK,
    'axes.facecolor':    FINVEE_CARD,
    'axes.edgecolor':    FINVEE_MUTED,
    'axes.labelcolor':   FINVEE_TEXT,
    'xtick.color':       FINVEE_MUTED,
    'ytick.color':       FINVEE_MUTED,
    'text.color':        FINVEE_TEXT,
    'grid.color':        '#21262D',
    'grid.linestyle':    '--',
    'grid.alpha':        0.6,
    'font.family':       'DejaVu Sans',
    'font.size':         11,
    'axes.titlesize':    13,
    'axes.titleweight':  'bold',
    'axes.titlepad':     12,
    'legend.facecolor':  FINVEE_CARD,
    'legend.edgecolor':  FINVEE_MUTED,
    'legend.labelcolor': FINVEE_TEXT,
    'savefig.facecolor': FINVEE_DARK,
    'savefig.dpi':       150,
    'savefig.bbox':      'tight',
    'savefig.pad_inches': 0.3,
})

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

def save_fig(fig, name):
    path = os.path.join(OUTPUT_PATH, f"{name}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [SAVED] output/{name}.png")
    return path

def header(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")

# ─────────────────────────────────────────────
# 1. DOWNLOAD DATASET FROM KAGGLE
# ─────────────────────────────────────────────
header("STEP 1: Downloading IEEE-CIS Dataset from Kaggle")

if not os.path.exists(os.path.join(DATA_PATH, 'train_transaction.csv')):
    print("\n[CMD] Running Kaggle CLI to download data...")
    if not os.path.exists(os.path.expanduser('~/.kaggle/kaggle.json')):
        print("\n[ERROR] kaggle.json not found. See prerequisites in script docstring.")
        exit(1)
    os.system("sudo pip3 install kaggle --quiet")
    cmd = (
        f"kaggle competitions download -c ieee-fraud-detection -p {DATA_PATH} --force && "
        f"unzip -o {DATA_PATH}/ieee-fraud-detection.zip -d {DATA_PATH} && "
        f"rm {DATA_PATH}/ieee-fraud-detection.zip"
    )
    if os.system(cmd) != 0:
        print("\n[ERROR] Download failed. Check API key and competition rules acceptance.")
        exit(1)
    print("\n[OK] Dataset downloaded and unzipped.")
else:
    print("\n[OK] Dataset already exists. Skipping download.")

# ─────────────────────────────────────────────
# 2. LOAD & PREPROCESS DATA
# ─────────────────────────────────────────────
header("STEP 2: Loading and Preprocessing Data")

def reduce_mem(df):
    for col in df.columns:
        ct = df[col].dtype
        if ct in ['int64', 'int32']:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif ct in ['float64', 'float32']:
            df[col] = pd.to_numeric(df[col], downcast='float')
    return df

print("\n[LOAD] Reading CSVs...")
train_txn = pd.read_csv(os.path.join(DATA_PATH, 'train_transaction.csv'))
train_id  = pd.read_csv(os.path.join(DATA_PATH, 'train_identity.csv'))
train_df  = pd.merge(train_txn, train_id, on='TransactionID', how='left')
del train_txn, train_id
gc.collect()

train_df = train_df.sample(frac=SUBSAMPLE_FRAC, random_state=42).reset_index(drop=True)
train_df = reduce_mem(train_df)

total_txn  = len(train_df)
fraud_cnt  = int(train_df['isFraud'].sum())
normal_cnt = total_txn - fraud_cnt
fraud_rate = fraud_cnt / total_txn

print(f"\n  Total transactions : {total_txn:,}")
print(f"  Fraud              : {fraud_cnt:,}  ({fraud_rate:.2%})")
print(f"  Normal             : {normal_cnt:,}  ({1-fraud_rate:.2%})")

# ── PLOT 1: Dataset Overview ─────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("AEGIS SME — IEEE-CIS Dataset Overview", fontsize=16, fontweight='bold', color=FINVEE_ACCENT)

# Class distribution
axes[0].bar(['Normal', 'Fraud'], [normal_cnt, fraud_cnt],
            color=[FINVEE_GREEN, FINVEE_RED], edgecolor='white', linewidth=0.5)
axes[0].set_title("Class Distribution")
axes[0].set_ylabel("Count")
for i, v in enumerate([normal_cnt, fraud_cnt]):
    axes[0].text(i, v + total_txn * 0.01, f"{v:,}\n({v/total_txn:.1%})",
                 ha='center', va='bottom', fontsize=10, color=FINVEE_TEXT)

# Transaction amount distribution
amt_normal = train_df[train_df['isFraud'] == 0]['TransactionAmt'].clip(upper=2000)
amt_fraud  = train_df[train_df['isFraud'] == 1]['TransactionAmt'].clip(upper=2000)
axes[1].hist(amt_normal, bins=60, alpha=0.6, color=FINVEE_GREEN, label='Normal', density=True)
axes[1].hist(amt_fraud,  bins=60, alpha=0.6, color=FINVEE_RED,   label='Fraud',  density=True)
axes[1].set_title("Transaction Amount Distribution")
axes[1].set_xlabel("Amount (USD, clipped at 2000)")
axes[1].set_ylabel("Density")
axes[1].legend()

# ProductCD distribution
prod_counts = train_df.groupby(['ProductCD', 'isFraud']).size().unstack(fill_value=0)
prod_counts.columns = ['Normal', 'Fraud']
prod_counts['FraudRate'] = prod_counts['Fraud'] / (prod_counts['Normal'] + prod_counts['Fraud'])
bars = axes[2].bar(prod_counts.index, prod_counts['FraudRate'] * 100,
                   color=FINVEE_ACCENT, edgecolor='white', linewidth=0.5)
axes[2].set_title("Fraud Rate by Product Category")
axes[2].set_xlabel("Product Code")
axes[2].set_ylabel("Fraud Rate (%)")
for bar, val in zip(bars, prod_counts['FraudRate']):
    axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 f"{val:.1%}", ha='center', va='bottom', fontsize=9, color=FINVEE_TEXT)

plt.tight_layout()
save_fig(fig, "01_dataset_overview")

# Feature selection & encoding
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
features_to_use = [f for f in USEFUL_FEATURES if f in train_df.columns]
X = train_df[features_to_use].copy()
y = train_df['isFraud'].copy()
del train_df
gc.collect()

label_encoders = {}
for col in X.select_dtypes(include=['object']).columns:
    X[col] = X[col].astype(str)
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

X = X.fillna(-999)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"\n  Train set: {len(X_train):,} | Test set: {len(X_test):,}")
print(f"  Features : {len(features_to_use)}")

# ─────────────────────────────────────────────
# 3. TRAIN LIGHTGBM
# ─────────────────────────────────────────────
header("STEP 3a: Training LightGBM Classifier")

lgb_train_aucs = []
lgb_val_aucs   = []

class AUCCallback:
    def __call__(self, env):
        if env.iteration % 50 == 0 or env.iteration == env.end_iteration - 1:
            for item in env.evaluation_result_list:
                if 'valid' in item[0]:
                    lgb_val_aucs.append((env.iteration, item[2]))
                elif 'train' in item[0]:
                    lgb_train_aucs.append((env.iteration, item[2]))

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'n_estimators': 1000,
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
    eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
    eval_names=['train', 'valid'],
    callbacks=[
        lgb.early_stopping(100, verbose=False),
        lgb.log_evaluation(100),
        lgb.record_evaluation(evals_result := {})
    ]
)

lgb_proba = lgb_model.predict_proba(X_test_scaled)[:, 1]
lgb_pred  = (lgb_proba > 0.5).astype(int)
lgb_auc   = roc_auc_score(y_test, lgb_proba)
lgb_ap    = average_precision_score(y_test, lgb_proba)
lgb_f1    = f1_score(y_test, lgb_pred)
lgb_prec  = precision_score(y_test, lgb_pred)
lgb_rec   = recall_score(y_test, lgb_pred)

print(f"\n  AUC-ROC            : {lgb_auc:.4f}")
print(f"  Avg Precision (AP) : {lgb_ap:.4f}")
print(f"  F1 Score           : {lgb_f1:.4f}")
print(f"  Precision          : {lgb_prec:.4f}")
print(f"  Recall             : {lgb_rec:.4f}")

# ── PLOT 2: LightGBM Training Curves ─────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("LightGBM — Training Progress", fontsize=16, fontweight='bold', color=FINVEE_ACCENT)

train_auc_vals = evals_result.get('train', {}).get('auc', [])
valid_auc_vals = evals_result.get('valid', {}).get('auc', [])
iters = list(range(len(train_auc_vals)))

axes[0].plot(iters, train_auc_vals, color=FINVEE_GREEN, linewidth=2, label='Train AUC')
axes[0].plot(iters, valid_auc_vals, color=FINVEE_ACCENT, linewidth=2, label='Validation AUC')
axes[0].axhline(y=lgb_auc, color=FINVEE_RED, linestyle='--', linewidth=1.5, alpha=0.8,
                label=f'Best Val AUC: {lgb_auc:.4f}')
axes[0].set_title("AUC-ROC per Iteration")
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("AUC-ROC")
axes[0].legend()
axes[0].grid(True)

# Feature importance (top 20)
feat_imp = pd.Series(lgb_model.feature_importances_, index=features_to_use).nlargest(20)
colors = [FINVEE_ACCENT if i < 5 else FINVEE_MUTED for i in range(len(feat_imp))]
axes[1].barh(feat_imp.index[::-1], feat_imp.values[::-1], color=colors[::-1], edgecolor='none')
axes[1].set_title("Top 20 Feature Importances")
axes[1].set_xlabel("Importance Score")
axes[1].grid(True, axis='x')

plt.tight_layout()
save_fig(fig, "02_lgbm_training_curves")

# ── PLOT 3: LightGBM Evaluation ──────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("LightGBM — Model Evaluation", fontsize=16, fontweight='bold', color=FINVEE_ACCENT)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, lgb_proba)
axes[0].plot(fpr, tpr, color=FINVEE_ACCENT, linewidth=2.5, label=f'AUC = {lgb_auc:.4f}')
axes[0].plot([0, 1], [0, 1], color=FINVEE_MUTED, linestyle='--', linewidth=1.5, label='Random')
axes[0].fill_between(fpr, tpr, alpha=0.15, color=FINVEE_ACCENT)
axes[0].set_title("ROC Curve")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend()
axes[0].grid(True)

# Precision-Recall Curve
prec_vals, rec_vals, _ = precision_recall_curve(y_test, lgb_proba)
axes[1].plot(rec_vals, prec_vals, color=FINVEE_GREEN, linewidth=2.5, label=f'AP = {lgb_ap:.4f}')
axes[1].axhline(y=fraud_rate, color=FINVEE_MUTED, linestyle='--', linewidth=1.5,
                label=f'Baseline ({fraud_rate:.2%})')
axes[1].fill_between(rec_vals, prec_vals, alpha=0.15, color=FINVEE_GREEN)
axes[1].set_title("Precision-Recall Curve")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].legend()
axes[1].grid(True)

# Confusion Matrix
cm = confusion_matrix(y_test, lgb_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2],
            xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'],
            linewidths=0.5, linecolor=FINVEE_DARK,
            annot_kws={'size': 14, 'weight': 'bold', 'color': 'white'})
axes[2].set_title("Confusion Matrix")
axes[2].set_xlabel("Predicted")
axes[2].set_ylabel("Actual")

plt.tight_layout()
save_fig(fig, "03_lgbm_evaluation")

# ── PLOT 4: Risk Score Distribution ──────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("LightGBM — Risk Score Distribution", fontsize=16, fontweight='bold', color=FINVEE_ACCENT)

axes[0].hist(lgb_proba[y_test == 0], bins=80, alpha=0.7, color=FINVEE_GREEN,
             label='Normal', density=True)
axes[0].hist(lgb_proba[y_test == 1], bins=80, alpha=0.7, color=FINVEE_RED,
             label='Fraud', density=True)
axes[0].axvline(x=0.45, color=FINVEE_YELLOW, linestyle='--', linewidth=2,
                label='Step-Up Auth (0.45)')
axes[0].axvline(x=0.75, color=FINVEE_RED, linestyle='--', linewidth=2,
                label='Block (0.75)')
axes[0].set_title("Risk Score Distribution (Full Range)")
axes[0].set_xlabel("Risk Score")
axes[0].set_ylabel("Density")
axes[0].legend()
axes[0].grid(True)

# Zoomed in on high-risk range
axes[1].hist(lgb_proba[(y_test == 0) & (lgb_proba > 0.3)], bins=60, alpha=0.7,
             color=FINVEE_GREEN, label='Normal', density=True)
axes[1].hist(lgb_proba[(y_test == 1) & (lgb_proba > 0.3)], bins=60, alpha=0.7,
             color=FINVEE_RED, label='Fraud', density=True)
axes[1].axvline(x=0.45, color=FINVEE_YELLOW, linestyle='--', linewidth=2,
                label='Step-Up Auth (0.45)')
axes[1].axvline(x=0.75, color=FINVEE_RED, linestyle='--', linewidth=2,
                label='Block (0.75)')
axes[1].set_title("Risk Score Distribution (Zoomed: > 0.3)")
axes[1].set_xlabel("Risk Score")
axes[1].set_ylabel("Density")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
save_fig(fig, "04_risk_score_distribution")

# ─────────────────────────────────────────────
# 4. TRAIN AUTOENCODER
# ─────────────────────────────────────────────
header("STEP 3b: Training Autoencoder (Anomaly Detection)")

X_normal_train = X_train_scaled[y_train.values == 0]
input_dim = X_normal_train.shape[1]

autoencoder = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(input_dim,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(input_dim, activation='linear'),
])
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')

history = autoencoder.fit(
    X_normal_train, X_normal_train,
    epochs=150,
    batch_size=512,
    validation_split=0.1,
    verbose=0,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True,
                                          monitor='val_loss'),
        tf.keras.callbacks.ReduceLROnPlateau(patience=8, factor=0.5, verbose=0)
    ]
)
print(f"\n  Epochs trained     : {len(history.history['loss'])}")
print(f"  Final train loss   : {history.history['loss'][-1]:.6f}")
print(f"  Final val loss     : {history.history['val_loss'][-1]:.6f}")

# Threshold
X_normal_recon = autoencoder.predict(X_normal_train, verbose=0, batch_size=2048)
normal_errors  = np.mean(np.square(X_normal_train - X_normal_recon), axis=1)
ae_threshold   = np.percentile(normal_errors, 98)

# Evaluate
X_test_recon = autoencoder.predict(X_test_scaled, verbose=0, batch_size=2048)
ae_errors    = np.mean(np.square(X_test_scaled - X_test_recon), axis=1)
ae_auc       = roc_auc_score(y_test, ae_errors)
ae_pred      = (ae_errors > ae_threshold).astype(int)
ae_f1        = f1_score(y_test, ae_pred)

print(f"  Anomaly threshold  : {ae_threshold:.6f}")
print(f"  Autoencoder AUC    : {ae_auc:.4f}")
print(f"  Autoencoder F1     : {ae_f1:.4f}")

# ── PLOT 5: Autoencoder Training ─────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("Autoencoder — Training & Anomaly Detection", fontsize=16, fontweight='bold', color=FINVEE_ACCENT)

# Loss curves
epochs = range(1, len(history.history['loss']) + 1)
axes[0].plot(epochs, history.history['loss'], color=FINVEE_GREEN, linewidth=2, label='Train Loss')
axes[0].plot(epochs, history.history['val_loss'], color=FINVEE_ACCENT, linewidth=2, label='Val Loss')
axes[0].set_title("Training Loss (MSE)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE Loss")
axes[0].legend()
axes[0].grid(True)

# Reconstruction error distribution
err_normal = ae_errors[y_test.values == 0]
err_fraud  = ae_errors[y_test.values == 1]
clip_val   = np.percentile(ae_errors, 99.5)
axes[1].hist(np.clip(err_normal, 0, clip_val), bins=80, alpha=0.7,
             color=FINVEE_GREEN, label='Normal', density=True)
axes[1].hist(np.clip(err_fraud, 0, clip_val), bins=80, alpha=0.7,
             color=FINVEE_RED, label='Fraud', density=True)
axes[1].axvline(x=ae_threshold, color=FINVEE_YELLOW, linestyle='--', linewidth=2,
                label=f'Threshold: {ae_threshold:.4f}')
axes[1].set_title("Reconstruction Error Distribution")
axes[1].set_xlabel("Reconstruction Error (MSE)")
axes[1].set_ylabel("Density")
axes[1].legend()
axes[1].grid(True)

# Autoencoder ROC Curve
fpr_ae, tpr_ae, _ = roc_curve(y_test, ae_errors)
axes[2].plot(fpr_ae, tpr_ae, color=FINVEE_ACCENT, linewidth=2.5, label=f'AUC = {ae_auc:.4f}')
axes[2].plot([0, 1], [0, 1], color=FINVEE_MUTED, linestyle='--', linewidth=1.5, label='Random')
axes[2].fill_between(fpr_ae, tpr_ae, alpha=0.15, color=FINVEE_ACCENT)
axes[2].set_title("Autoencoder ROC Curve")
axes[2].set_xlabel("False Positive Rate")
axes[2].set_ylabel("True Positive Rate")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
save_fig(fig, "05_autoencoder_evaluation")

# ─────────────────────────────────────────────
# 5. ENSEMBLE ANALYSIS
# ─────────────────────────────────────────────
header("STEP 4: Ensemble Analysis (LightGBM + Autoencoder)")

# Normalize AE errors to [0, 1]
ae_norm = (ae_errors - ae_errors.min()) / (ae_errors.max() - ae_errors.min() + 1e-9)
ensemble_score = 0.7 * lgb_proba + 0.3 * ae_norm
ensemble_pred  = (ensemble_score > 0.5).astype(int)
ens_auc  = roc_auc_score(y_test, ensemble_score)
ens_ap   = average_precision_score(y_test, ensemble_score)
ens_f1   = f1_score(y_test, ensemble_pred)
ens_prec = precision_score(y_test, ensemble_pred)
ens_rec  = recall_score(y_test, ensemble_pred)

print(f"\n  Ensemble AUC-ROC   : {ens_auc:.4f}")
print(f"  Ensemble Avg Prec  : {ens_ap:.4f}")
print(f"  Ensemble F1        : {ens_f1:.4f}")
print(f"  Ensemble Precision : {ens_prec:.4f}")
print(f"  Ensemble Recall    : {ens_rec:.4f}")

# ── PLOT 6: Model Comparison ──────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Model Comparison — LightGBM vs Autoencoder vs Ensemble",
             fontsize=15, fontweight='bold', color=FINVEE_ACCENT)

# ROC curves comparison
fpr_ens, tpr_ens, _ = roc_curve(y_test, ensemble_score)
axes[0].plot(fpr, tpr, color=FINVEE_GREEN, linewidth=2.5, label=f'LightGBM (AUC={lgb_auc:.4f})')
axes[0].plot(fpr_ae, tpr_ae, color=FINVEE_YELLOW, linewidth=2.5, label=f'Autoencoder (AUC={ae_auc:.4f})')
axes[0].plot(fpr_ens, tpr_ens, color=FINVEE_ACCENT, linewidth=2.5,
             linestyle='--', label=f'Ensemble (AUC={ens_auc:.4f})')
axes[0].plot([0, 1], [0, 1], color=FINVEE_MUTED, linestyle=':', linewidth=1.5, label='Random')
axes[0].set_title("ROC Curve Comparison")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend()
axes[0].grid(True)

# Metrics bar chart
metrics = ['AUC-ROC', 'Avg Precision', 'F1 Score', 'Precision', 'Recall']
lgb_vals = [lgb_auc, lgb_ap, lgb_f1, lgb_prec, lgb_rec]
ens_vals = [ens_auc, ens_ap, ens_f1, ens_prec, ens_rec]
x = np.arange(len(metrics))
w = 0.35
bars1 = axes[1].bar(x - w/2, lgb_vals, w, label='LightGBM', color=FINVEE_GREEN, alpha=0.85)
bars2 = axes[1].bar(x + w/2, ens_vals, w, label='Ensemble', color=FINVEE_ACCENT, alpha=0.85)
for bar in list(bars1) + list(bars2):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8.5,
                 color=FINVEE_TEXT)
axes[1].set_title("Metrics Comparison")
axes[1].set_xticks(x)
axes[1].set_xticklabels(metrics, rotation=15, ha='right')
axes[1].set_ylim(0, 1.12)
axes[1].legend()
axes[1].grid(True, axis='y')

plt.tight_layout()
save_fig(fig, "06_model_comparison")

# ── PLOT 7: Final Summary Card ────────────────
fig = plt.figure(figsize=(16, 9))
fig.patch.set_facecolor(FINVEE_DARK)
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.4)
fig.suptitle("⚔️  AEGIS SME — Training Summary Report\nTeam Finvee | Varsity Hackathon 2026",
             fontsize=18, fontweight='bold', color=FINVEE_ACCENT, y=0.98)

def metric_card(ax, title, value, subtitle="", color=FINVEE_ACCENT):
    ax.set_facecolor(FINVEE_CARD)
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 0.65, value, ha='center', va='center', fontsize=26,
            fontweight='bold', color=color, transform=ax.transAxes)
    ax.text(0.5, 0.25, title, ha='center', va='center', fontsize=10,
            color=FINVEE_TEXT, transform=ax.transAxes)
    if subtitle:
        ax.text(0.5, 0.08, subtitle, ha='center', va='center', fontsize=8,
                color=FINVEE_MUTED, transform=ax.transAxes)

ax1 = fig.add_subplot(gs[0, 0])
metric_card(ax1, "Ensemble AUC-ROC", f"{ens_auc:.4f}", "Primary metric", FINVEE_ACCENT)

ax2 = fig.add_subplot(gs[0, 1])
metric_card(ax2, "LightGBM AUC", f"{lgb_auc:.4f}", "Supervised model", FINVEE_GREEN)

ax3 = fig.add_subplot(gs[0, 2])
metric_card(ax3, "Autoencoder AUC", f"{ae_auc:.4f}", "Unsupervised model", FINVEE_YELLOW)

ax4 = fig.add_subplot(gs[0, 3])
metric_card(ax4, "Ensemble F1", f"{ens_f1:.4f}", "Harmonic mean", FINVEE_RED)

ax5 = fig.add_subplot(gs[1, 0])
metric_card(ax5, "Training Samples", f"{len(X_train):,}", f"{SUBSAMPLE_FRAC:.0%} of dataset", FINVEE_MUTED)

ax6 = fig.add_subplot(gs[1, 1])
metric_card(ax6, "Features Used", f"{len(features_to_use)}", "IEEE-CIS features", FINVEE_MUTED)

ax7 = fig.add_subplot(gs[1, 2])
metric_card(ax7, "Fraud Rate", f"{fraud_rate:.2%}", "Class imbalance", FINVEE_RED)

ax8 = fig.add_subplot(gs[1, 3])
metric_card(ax8, "AE Epochs", f"{len(history.history['loss'])}", "Early stopped", FINVEE_MUTED)

# Pipeline flow
ax_pipe = fig.add_subplot(gs[2, :])
ax_pipe.set_facecolor(FINVEE_CARD)
ax_pipe.set_xlim(0, 10)
ax_pipe.set_ylim(0, 1)
ax_pipe.set_xticks([])
ax_pipe.set_yticks([])
for spine in ax_pipe.spines.values():
    spine.set_edgecolor(FINVEE_ACCENT)
    spine.set_linewidth(1.5)

steps = [
    ("IEEE-CIS\nDataset", 0.5, FINVEE_MUTED),
    ("Feature\nEngineering", 1.8, FINVEE_MUTED),
    ("LightGBM\nClassifier", 3.3, FINVEE_GREEN),
    ("Autoencoder\nAnomaly", 4.8, FINVEE_YELLOW),
    ("Ensemble\nScore", 6.3, FINVEE_ACCENT),
    ("AEGIS\nDecision", 7.8, FINVEE_RED),
    ("Agent\nNotification", 9.3, FINVEE_ACCENT),
]
for label, x, color in steps:
    ax_pipe.add_patch(plt.FancyBboxPatch((x - 0.55, 0.15), 1.1, 0.7,
                                          boxstyle="round,pad=0.05",
                                          facecolor=color, alpha=0.2,
                                          edgecolor=color, linewidth=1.5))
    ax_pipe.text(x, 0.5, label, ha='center', va='center',
                 fontsize=8.5, fontweight='bold', color=color)
    if x < 9.3:
        ax_pipe.annotate('', xy=(x + 0.65, 0.5), xytext=(x + 0.55, 0.5),
                         arrowprops=dict(arrowstyle='->', color=FINVEE_MUTED, lw=1.5))

ax_pipe.set_title("AEGIS SME Full Pipeline", color=FINVEE_TEXT, pad=8)

plt.tight_layout(rect=[0, 0, 1, 0.95])
save_fig(fig, "07_training_summary_card")

# ─────────────────────────────────────────────
# 6. SAVE ALL ARTIFACTS
# ─────────────────────────────────────────────
header("STEP 5: Saving Model Artifacts")

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
    "train_samples": len(X_train),
    "test_samples": len(X_test),
    "fraud_rate": float(fraud_rate),
    "features": features_to_use,
    "ae_threshold": float(ae_threshold),
    "lgb_auc": float(lgb_auc),
    "lgb_ap": float(lgb_ap),
    "lgb_f1": float(lgb_f1),
    "ae_auc": float(ae_auc),
    "ae_f1": float(ae_f1),
    "ensemble_auc": float(ens_auc),
    "ensemble_ap": float(ens_ap),
    "ensemble_f1": float(ens_f1),
    "timestamp": TIMESTAMP,
}
with open(os.path.join(MODEL_PATH, "metadata_ieee.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n  [OK] lgb_model_ieee.pkl")
print(f"  [OK] autoencoder_ieee.keras")
print(f"  [OK] scaler_ieee.pkl")
print(f"  [OK] label_encoders_ieee.pkl")
print(f"  [OK] metadata_ieee.json")

# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print(f"""
{'=' * 60}
  AEGIS SME — Retraining Complete!
  Team Finvee | {TIMESTAMP}
{'=' * 60}

  MODELS SAVED  →  {MODEL_PATH}
  VISUALIZATIONS →  {OUTPUT_PATH}

  ┌─────────────────────────────────────────┐
  │  MODEL PERFORMANCE SUMMARY             │
  ├─────────────────────────────────────────┤
  │  LightGBM AUC-ROC    : {lgb_auc:.4f}          │
  │  LightGBM F1         : {lgb_f1:.4f}          │
  │  Autoencoder AUC     : {ae_auc:.4f}          │
  │  Ensemble AUC-ROC    : {ens_auc:.4f}          │
  │  Ensemble F1         : {ens_f1:.4f}          │
  └─────────────────────────────────────────┘

  OUTPUT IMAGES GENERATED:
  01_dataset_overview.png
  02_lgbm_training_curves.png
  03_lgbm_evaluation.png
  04_risk_score_distribution.png
  05_autoencoder_evaluation.png
  06_model_comparison.png
  07_training_summary_card.png

{'=' * 60}
""")
