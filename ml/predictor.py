"""
AEGIS SME — ML Predictor (Inference Engine)
Team Finvee | Varsity Hackathon 2026
Loads trained models and provides fast inference for agent use.
"""

import numpy as np
import pickle
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

_lgb_model = None
_autoencoder = None
_scaler = None
_label_encoders = None
_metadata = None

def _load_models():
    global _lgb_model, _autoencoder, _scaler, _label_encoders, _metadata
    if _lgb_model is None:
        base = "/home/ubuntu/aegis-sme/models"
        with open(f"{base}/lgb_model.pkl", "rb") as f:
            _lgb_model = pickle.load(f)
        _autoencoder = tf.keras.models.load_model(f"{base}/autoencoder.keras")
        with open(f"{base}/scaler.pkl", "rb") as f:
            _scaler = pickle.load(f)
        with open(f"{base}/label_encoders.pkl", "rb") as f:
            _label_encoders = pickle.load(f)
        with open(f"{base}/metadata.json", "r") as f:
            _metadata = json.load(f)

def encode_transaction(txn: dict) -> np.ndarray:
    """Convert a transaction dict to feature vector."""
    _load_models()

    merchant_type = txn.get("merchant_type", "food")
    location = txn.get("location", "Jakarta")

    # Handle unseen labels gracefully
    le_mt = _label_encoders['merchant_type']
    le_loc = _label_encoders['location']
    mt_enc = le_mt.transform([merchant_type])[0] if merchant_type in le_mt.classes_ else 0
    loc_enc = le_loc.transform([location])[0] if location in le_loc.classes_ else 0

    features = np.array([[
        txn.get("hour", 12),
        txn.get("day_of_week", 1),
        txn.get("amount", 100000),
        txn.get("is_new_device", 0),
        txn.get("transaction_count_1h", 3),
        txn.get("transaction_count_24h", 15),
        txn.get("amount_vs_avg_ratio", 1.0),
        txn.get("location_mismatch", 0),
        mt_enc,
        loc_enc
    ]])
    return features

def predict(txn: dict) -> dict:
    """
    Run full ensemble prediction on a transaction.
    Returns: risk_score, ae_score, ensemble_score, decision, risk_level, explanation
    """
    _load_models()

    features = encode_transaction(txn)
    features_scaled = _scaler.transform(features)

    # LightGBM score
    lgb_score = float(_lgb_model.predict_proba(features_scaled)[0][1])

    # Autoencoder anomaly score
    recon = _autoencoder.predict(features_scaled, verbose=0)
    ae_error = float(np.mean(np.square(features_scaled - recon)))
    ae_threshold = _metadata["ae_threshold"]
    ae_score = min(ae_error / (ae_threshold * 3), 1.0)

    # Ensemble
    lgb_w = _metadata["lgb_weight"]
    ae_w = _metadata["ae_weight"]
    ensemble = lgb_w * lgb_score + ae_w * ae_score

    # Decision
    thresholds = _metadata["decision_thresholds"]
    if ensemble >= thresholds["block"]:
        decision, risk_level = "BLOCK", "HIGH"
    elif ensemble >= thresholds["step_up"]:
        decision, risk_level = "STEP_UP_AUTH", "MEDIUM"
    else:
        decision, risk_level = "APPROVE", "LOW"

    # Build explanation factors
    factors = []
    if txn.get("is_new_device", 0):
        factors.append("perangkat baru tidak dikenal")
    if txn.get("location_mismatch", 0):
        factors.append("lokasi tidak sesuai riwayat")
    if txn.get("amount_vs_avg_ratio", 1.0) > 5:
        factors.append(f"jumlah {txn.get('amount_vs_avg_ratio',1):.1f}x lebih besar dari rata-rata")
    if txn.get("transaction_count_1h", 0) > 10:
        factors.append("frekuensi transaksi sangat tinggi dalam 1 jam")
    if txn.get("hour", 12) in [0, 1, 2, 3, 4]:
        factors.append("transaksi di jam tidak wajar (dini hari)")

    explanation = "Faktor risiko: " + "; ".join(factors) if factors else "Tidak ada faktor risiko signifikan"

    return {
        "lgb_score": round(lgb_score, 4),
        "ae_score": round(ae_score, 4),
        "ae_error": round(ae_error, 6),
        "ensemble_score": round(ensemble, 4),
        "decision": decision,
        "risk_level": risk_level,
        "explanation": explanation,
        "factors": factors
    }

if __name__ == "__main__":
    # Quick test
    test_txn = {
        "transaction_id": "TXN999999",
        "merchant_id": "UKM001",
        "merchant_type": "food",
        "amount": 8500000,
        "hour": 2,
        "day_of_week": 1,
        "location": "Jakarta",
        "is_new_device": 1,
        "transaction_count_1h": 1,
        "transaction_count_24h": 5,
        "amount_vs_avg_ratio": 12.5,
        "location_mismatch": 1
    }
    result = predict(test_txn)
    print("Test Prediction:")
    for k, v in result.items():
        print(f"  {k}: {v}")
