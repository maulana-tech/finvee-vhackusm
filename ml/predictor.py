"""
AEGIS SME — IEEE-CIS Predictor (v2)
=====================================
Team Finvee | Varsity Hackathon 2026 | Case Study 2

Loads the real IEEE-CIS trained models:
  - LightGBM Classifier (AUC ~0.952)
  - Autoencoder Anomaly Detector
  - StandardScaler + LabelEncoders

Exposes predict() and get_model_info() for agent and API use.
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR   = os.path.join(BASE_DIR, "models_ieee")

LGB_PATH    = os.path.join(MODEL_DIR, "lgb_model_ieee.pkl")
AE_PATH     = os.path.join(MODEL_DIR, "autoencoder_ieee.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_ieee.pkl")
LE_PATH     = os.path.join(MODEL_DIR, "label_encoders_ieee.pkl")
META_PATH   = os.path.join(MODEL_DIR, "metadata_ieee.json")

# ── IEEE-CIS Feature List (60 features — must match training order) ───────────
IEEE_FEATURES = [
    "TransactionAmt", "ProductCD",
    "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "dist1", "dist2",
    "P_emaildomain", "R_emaildomain",
    "C1","C2","C4","C5","C6","C7","C8","C9","C10","C11","C12","C13","C14",
    "D1","D2","D3","D4","D5","D10","D15",
    "M1","M2","M3","M4","M5","M6","M7","M8","M9",
    "id_01","id_02","id_03","id_04","id_05","id_06","id_11",
    "id_12","id_13","id_15","id_17","id_19","id_20",
    "id_30","id_31","id_38",
    "DeviceType","DeviceInfo"
]

CATEGORICAL_FEATURES = [
    "ProductCD","card4","card6","P_emaildomain","R_emaildomain",
    "M1","M2","M3","M4","M5","M6","M7","M8","M9",
    "id_12","id_15","id_30","id_31","id_38",
    "DeviceType","DeviceInfo"
]

# ── Lazy-loaded globals ───────────────────────────────────────────────────────
_lgb_model      = None
_autoencoder    = None
_scaler         = None
_label_encoders = None
_metadata       = None
_ae_threshold   = None
_models_loaded  = False


def _load_models():
    """Load all model artifacts once (lazy loading)."""
    global _lgb_model, _autoencoder, _scaler, _label_encoders
    global _metadata, _ae_threshold, _models_loaded

    if _models_loaded:
        return

    print("🔄 Loading IEEE-CIS model artifacts...")

    with open(LGB_PATH, "rb") as f:
        _lgb_model = pickle.load(f)

    with open(SCALER_PATH, "rb") as f:
        _scaler = pickle.load(f)

    with open(LE_PATH, "rb") as f:
        _label_encoders = pickle.load(f)

    with open(META_PATH, "r") as f:
        _metadata = json.load(f)

    _ae_threshold = _metadata["ae_threshold"]

    try:
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        _autoencoder = tf.keras.models.load_model(AE_PATH)
    except Exception as e:
        print(f"⚠️  Autoencoder load warning: {e} — using LightGBM only")
        _autoencoder = None

    _models_loaded = True
    print(f"✅ Models loaded! LGB AUC={_metadata['lgb_auc']:.4f} | Ensemble AUC={_metadata['ensemble_auc']:.4f}")


def _encode_transaction(txn: dict) -> np.ndarray:
    """
    Map a simplified transaction dict to the 60-feature IEEE-CIS vector.

    Supports both simplified keys (amount, merchant_type, is_new_device, etc.)
    and raw IEEE-CIS keys (TransactionAmt, card1, etc.).
    Missing values default to -999 (same as training preprocessing).
    """
    row = {}

    # ── Numeric fields ────────────────────────────────────────────────────────
    row["TransactionAmt"] = float(txn.get("amount", txn.get("TransactionAmt", 100.0)))
    row["card1"]  = float(txn.get("card1",  9500))
    row["card2"]  = float(txn.get("card2",  321.0))
    row["card3"]  = float(txn.get("card3",  150.0))
    row["card5"]  = float(txn.get("card5",  226.0))
    row["addr1"]  = float(txn.get("addr1",  299.0))
    row["addr2"]  = float(txn.get("addr2",  87.0))
    row["dist1"]  = float(txn.get("dist1",  -999))
    row["dist2"]  = float(txn.get("dist2",  -999))

    # C-features: velocity / count signals — mapped from behavioral inputs
    txn_1h    = float(txn.get("transaction_count_1h",  1.0))
    txn_24h   = float(txn.get("transaction_count_24h", 1.0))
    amt_ratio = float(txn.get("amount_vs_avg_ratio", 1.0))
    is_new    = 1.0 if txn.get("is_new_device", False) else 0.0
    loc_mis   = 1.0 if txn.get("location_mismatch", False) else 0.0

    # C1: count of addr1 per card — high velocity = suspicious
    row["C1"]  = float(txn.get("C1",  txn_24h * (1 + loc_mis * 3)))
    # C2: count of card per addr1
    row["C2"]  = float(txn.get("C2",  txn_1h * (1 + is_new * 5)))
    # C4: count of card per P_emaildomain
    row["C4"]  = float(txn.get("C4",  is_new * txn_1h * 2))
    # C5: count of card per R_emaildomain
    row["C5"]  = float(txn.get("C5",  loc_mis * 3))
    # C6: count of addr1 per card (different angle)
    row["C6"]  = float(txn.get("C6",  txn_24h))
    # C7: count of card per addr1 (different angle)
    row["C7"]  = float(txn.get("C7",  is_new * txn_24h))
    # C8: count of card per P_emaildomain (different angle)
    row["C8"]  = float(txn.get("C8",  loc_mis * txn_1h))
    # C9: count of card per addr1 (1h window)
    row["C9"]  = float(txn.get("C9",  txn_1h * (1 + is_new * 3)))
    # C10: count of addr1 per card (1h window)
    row["C10"] = float(txn.get("C10", is_new * loc_mis * 5))
    # C11: count of card per addr1 (24h window)
    row["C11"] = float(txn.get("C11", txn_24h * (1 + loc_mis)))
    # C12: count of card per addr1 (velocity anomaly)
    row["C12"] = float(txn.get("C12", is_new * loc_mis * txn_1h))
    # C13: count of card per P_emaildomain (amount-weighted)
    row["C13"] = float(txn.get("C13", amt_ratio * (1 + is_new)))
    # C14: count of card per addr1 (long window)
    row["C14"] = float(txn.get("C14", txn_24h * (1 + is_new)))

    # D-features: time delta signals (days since last transaction)
    # D1: days since last transaction — new device = 0 (first time)
    row["D1"]  = float(txn.get("D1",  0.0 if is_new else 1.0))
    # D2: days since last transaction (card)
    row["D2"]  = float(txn.get("D2",  -999 if is_new else 1.0))
    # D3: days since last transaction (addr1)
    row["D3"]  = float(txn.get("D3",  -999 if loc_mis else 1.0))
    # D4: days since last transaction (P_emaildomain)
    row["D4"]  = float(txn.get("D4",  -999 if is_new else 2.0))
    # D5: days since last transaction (R_emaildomain)
    row["D5"]  = float(txn.get("D5",  -999))
    # D10: days since last transaction (device)
    row["D10"] = float(txn.get("D10", 0.0 if is_new else 5.0))
    # D15: days since last transaction (amount ratio proxy)
    row["D15"] = float(txn.get("D15", amt_ratio * (1 + is_new * 2)))

    # id numeric fields
    row["id_01"] = float(txn.get("id_01", 0.0))
    row["id_02"] = float(txn.get("id_02", -999))
    row["id_03"] = float(txn.get("id_03", -999))
    row["id_04"] = float(txn.get("id_04", -999))
    row["id_05"] = float(txn.get("id_05", 0.0))
    row["id_06"] = float(txn.get("id_06", 0.0))
    row["id_11"] = float(txn.get("id_11", -999))
    row["id_13"] = float(txn.get("id_13", -999))
    row["id_17"] = float(txn.get("id_17", -999))
    row["id_19"] = float(txn.get("id_19", -999))
    row["id_20"] = float(txn.get("id_20", -999))

    # ── Categorical fields ────────────────────────────────────────────────────
    # ProductCD: W=wallet/web, H=hotel, C=cash, R=retail, S=service
    merchant_type = str(txn.get("merchant_type", "W")).lower()
    product_map = {
        "wallet": "W", "web": "W", "online": "W", "w": "W",
        "hotel": "H", "travel": "H", "h": "H",
        "cash": "C", "atm": "C", "c": "C",
        "retail": "R", "store": "R", "r": "R",
        "service": "S", "s": "S",
    }
    row["ProductCD"] = product_map.get(merchant_type, "W")

    row["card4"] = txn.get("card4", "visa")
    row["card6"] = txn.get("card6", "debit")

    row["P_emaildomain"] = txn.get("P_emaildomain", txn.get("email_domain", "gmail.com"))
    row["R_emaildomain"] = txn.get("R_emaildomain", "nan")

    # M-features (match flags)
    is_new_device = bool(txn.get("is_new_device", False))
    loc_mismatch  = bool(txn.get("location_mismatch", False))
    row["M1"] = "T"
    row["M2"] = "T"
    row["M3"] = "T"
    row["M4"] = "M0"
    row["M5"] = "F" if is_new_device else "T"
    row["M6"] = "F" if loc_mismatch  else "T"
    row["M7"] = "nan"
    row["M8"] = "nan"
    row["M9"] = "nan"

    # id categorical
    row["id_12"] = "Found"
    row["id_15"] = "New" if is_new_device else "Found"
    row["id_30"] = txn.get("id_30", "Android")
    row["id_31"] = txn.get("id_31", "Generic/Android")
    row["id_38"] = "T"

    device_type = str(txn.get("device_type", "mobile")).lower()
    row["DeviceType"] = "mobile" if device_type in ["mobile", "phone", "smartphone"] else "desktop"
    row["DeviceInfo"] = txn.get("DeviceInfo", "nan")

    # ── Build DataFrame and label-encode ──────────────────────────────────────
    df = pd.DataFrame([row], columns=IEEE_FEATURES)

    for col in CATEGORICAL_FEATURES:
        if col in _label_encoders:
            le  = _label_encoders[col]
            val = str(df[col].iloc[0])
            if val in le.classes_:
                df[col] = le.transform([val])
            elif "nan" in le.classes_:
                df[col] = le.transform(["nan"])
            else:
                df[col] = 0
        else:
            df[col] = -999

    df = df.fillna(-999)
    return df.values.astype(np.float32)


def _compute_behavioral_score(txn: dict) -> float:
    """
    Rule-based behavioral risk score [0.0, 1.0].
    Captures velocity, device, location, and time anomalies that are
    well-understood fraud signals but hard to encode via IEEE-CIS features alone.
    """
    score = 0.0

    amount       = float(txn.get("amount", 0))
    amt_ratio    = float(txn.get("amount_vs_avg_ratio", 1.0))
    txn_1h       = int(txn.get("transaction_count_1h", 1))
    txn_24h      = int(txn.get("transaction_count_24h", 1))
    hour         = int(txn.get("hour", 12))
    is_new       = bool(txn.get("is_new_device", False))
    loc_mismatch = bool(txn.get("location_mismatch", False))

    # --- Device risk ---
    if is_new:
        score += 0.25  # New device is a strong fraud signal

    # --- Location risk ---
    if loc_mismatch:
        score += 0.20  # Location mismatch is a strong fraud signal

    # --- Combined device + location (Account Takeover pattern) ---
    if is_new and loc_mismatch:
        score += 0.15  # Extra boost for combined pattern

    # --- Amount anomaly ---
    if amt_ratio > 20:
        score += 0.20
    elif amt_ratio > 10:
        score += 0.15
    elif amt_ratio > 5:
        score += 0.08
    elif amt_ratio > 3:
        score += 0.03

    # --- Velocity anomaly (1h window) ---
    if txn_1h > 15:
        score += 0.15
    elif txn_1h > 10:
        score += 0.10
    elif txn_1h > 5:
        score += 0.05

    # --- Time anomaly (dini hari = off-hours) ---
    if hour in [0, 1, 2, 3, 4]:
        score += 0.08
        if is_new:
            score += 0.05  # Extra: new device at off-hours

    # --- High absolute amount ---
    if amount > 5000:
        score += 0.05
    elif amount > 2000:
        score += 0.02

    return min(score, 1.0)


def predict(txn: dict) -> dict:
    """
    Run full ensemble prediction on a transaction.

    Parameters
    ----------
    txn : dict — transaction data (simplified or raw IEEE-CIS keys)

    Returns
    -------
    dict with risk scores, decision, risk_level, and model metadata
    """
    _load_models()

    X        = _encode_transaction(txn)
    X_scaled = _scaler.transform(X)

    # LightGBM (IEEE-CIS trained, AUC=0.9522)
    lgb_score = float(_lgb_model.predict_proba(X_scaled)[0][1])

    # Autoencoder anomaly
    ae_score_norm = 0.0
    ae_score_raw  = 0.0
    if _autoencoder is not None:
        try:
            recon        = _autoencoder.predict(X_scaled, verbose=0)
            ae_score_raw = float(np.mean(np.square(X_scaled - recon)))
            ae_score_norm = min(ae_score_raw / (_ae_threshold + 1e-9), 1.0)
        except Exception:
            ae_score_norm = 0.0

    # Behavioral risk score (rule-based, captures signals LGB can't infer from simplified input)
    # This compensates for the fact that IEEE-CIS C/D features are pre-computed aggregates
    # that can't be perfectly reconstructed at inference time from simplified transaction inputs
    behavioral_score = _compute_behavioral_score(txn)

    # Hybrid ensemble:
    # - LightGBM: structural fraud patterns (card type, email domain, amount range)
    # - Behavioral: velocity, device, location anomalies
    # - AE: only if not saturated
    if ae_score_norm > 0.95:
        # AE saturated — use LGB + behavioral
        ensemble_score = 0.45 * lgb_score + 0.55 * behavioral_score
    else:
        ensemble_score = 0.40 * lgb_score + 0.45 * behavioral_score + 0.15 * ae_score_norm

    ensemble_score = min(ensemble_score, 1.0)

    # Decision thresholds (calibrated for hybrid score distribution)
    if ensemble_score >= 0.65:
        decision, risk_level = "BLOCK", "CRITICAL"
    elif ensemble_score >= 0.40:
        decision, risk_level = "STEP_UP_AUTH", "HIGH"
    elif ensemble_score >= 0.20:
        decision, risk_level = "STEP_UP_AUTH", "MEDIUM"
    else:
        decision, risk_level = "APPROVE", "LOW"

    # Confidence: distance from nearest threshold
    boundaries  = [0.0, 0.20, 0.45, 0.75, 1.0]
    min_dist    = min(abs(ensemble_score - b) for b in boundaries)
    confidence  = min(min_dist * 4 + 0.5, 1.0)

    # Human-readable risk factors
    factors = []
    if txn.get("is_new_device", False):
        factors.append("perangkat baru tidak dikenal")
    if txn.get("location_mismatch", False):
        factors.append("lokasi tidak sesuai riwayat")
    if float(txn.get("amount_vs_avg_ratio", 1.0)) > 5:
        factors.append(f"jumlah {txn.get('amount_vs_avg_ratio',1):.1f}x lebih besar dari rata-rata")
    if float(txn.get("transaction_count_1h", 0)) > 10:
        factors.append("frekuensi transaksi sangat tinggi dalam 1 jam")
    if int(txn.get("hour", 12)) in [0, 1, 2, 3, 4]:
        factors.append("transaksi di jam tidak wajar (dini hari)")
    if lgb_score > 0.7:
        factors.append(f"LightGBM mendeteksi pola fraud kuat (score={lgb_score:.3f})")
    if ae_score_norm > 0.5:
        factors.append(f"Autoencoder mendeteksi anomali perilaku (score={ae_score_norm:.3f})")

    explanation = "Faktor risiko: " + "; ".join(factors) if factors else "Tidak ada faktor risiko signifikan"

    return {
        "lgb_score":          round(lgb_score, 4),
        "ae_score":           round(ae_score_norm, 4),
        "ae_score_raw":       round(ae_score_raw, 6),
        "ensemble_score":     round(ensemble_score, 4),
        "decision":           decision,
        "risk_level":         risk_level,
        "confidence":         round(confidence, 4),
        "explanation":        explanation,
        "factors":            factors,
        "model_version":      "IEEE-CIS v1",
        "lgb_auc_ref":        _metadata["lgb_auc"],
        "ensemble_auc_ref":   _metadata["ensemble_auc"],
    }


def get_model_info() -> dict:
    """Return model metadata for display in dashboard/API."""
    _load_models()
    return {
        "dataset":          _metadata["dataset"],
        "train_samples":    _metadata["train_samples"],
        "fraud_rate":       _metadata["fraud_rate"],
        "lgb_auc":          _metadata["lgb_auc"],
        "lgb_f1":           _metadata["lgb_f1"],
        "ae_auc":           _metadata["ae_auc"],
        "ensemble_auc":     _metadata["ensemble_auc"],
        "ensemble_f1":      _metadata["ensemble_f1"],
        "features_count":   len(IEEE_FEATURES),
        "model_version":    "IEEE-CIS v1",
        "timestamp":        _metadata["timestamp"],
    }


if __name__ == "__main__":
    # Quick smoke test
    test_cases = [
        {
            "name": "Normal Transaction",
            "txn": {
                "amount": 150.0, "merchant_type": "W",
                "is_new_device": False, "location_mismatch": False,
                "transaction_count_1h": 1, "transaction_count_24h": 3,
                "amount_vs_avg_ratio": 1.2, "hour": 14,
                "card4": "visa", "card6": "debit",
            }
        },
        {
            "name": "Suspicious Transaction",
            "txn": {
                "amount": 9500.0, "merchant_type": "C",
                "is_new_device": True, "location_mismatch": True,
                "transaction_count_1h": 15, "transaction_count_24h": 40,
                "amount_vs_avg_ratio": 18.5, "hour": 2,
                "card4": "mastercard", "card6": "credit",
                "C1": 35, "C9": 15, "D15": 18.5,
            }
        },
    ]

    for case in test_cases:
        print(f"\n{'='*50}")
        print(f"Test: {case['name']}")
        result = predict(case["txn"])
        print(f"  Decision       : {result['decision']}")
        print(f"  Risk Level     : {result['risk_level']}")
        print(f"  LGB Score      : {result['lgb_score']}")
        print(f"  Ensemble Score : {result['ensemble_score']}")
        print(f"  Explanation    : {result['explanation']}")
