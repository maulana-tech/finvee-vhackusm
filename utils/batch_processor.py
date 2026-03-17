"""
AEGIS SME — Batch Processor
Team Finvee | Varsity Hackathon 2026

Handles CSV/Excel file upload, validation, and batch prediction.
"""

import io
import time
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple

# Required columns (minimum)
REQUIRED_COLS = ["amount", "hour"]

# All supported columns with defaults
COLUMN_DEFAULTS = {
    "transaction_id":        lambda i: f"TXN_UPLOAD_{i:04d}",
    "merchant_id":           "UKM001",
    "merchant_type":         "W",
    "amount":                100.0,
    "hour":                  12,
    "day_of_week":           1,
    "location":              "Jakarta",
    "device_id":             "DEV_0001",
    "is_new_device":         0,
    "transaction_count_1h":  2,
    "transaction_count_24h": 8,
    "amount_vs_avg_ratio":   1.0,
    "location_mismatch":     0,
}

# Indonesian city coordinates for mapping
CITY_COORDS = {
    "Jakarta":     (-6.2088,  106.8456),
    "Surabaya":    (-7.2575,  112.7521),
    "Bandung":     (-6.9175,  107.6191),
    "Medan":       (3.5952,   98.6722),
    "Solo":        (-7.5755,  110.8243),
    "Semarang":    (-6.9932,  110.4203),
    "Yogyakarta":  (-7.7956,  110.3695),
    "Makassar":    (-5.1477,  119.4327),
    "Palembang":   (-2.9761,  104.7754),
    "Denpasar":    (-8.6705,  115.2126),
    "Malang":      (-7.9797,  112.6304),
    "Tangerang":   (-6.1781,  106.6297),
    "Bekasi":      (-6.2349,  106.9896),
    "Depok":       (-6.4025,  106.7942),
    "Bogor":       (-6.5971,  106.8060),
    "Pekanbaru":   (0.5071,   101.4478),
    "Balikpapan":  (-1.2675,  116.8289),
    "Samarinda":   (-0.5022,  117.1536),
    "Pontianak":   (-0.0263,  109.3425),
    "Manado":      (1.4748,   124.8421),
    "Padang":      (-0.9471,  100.4172),
    "Banjarmasin": (-3.3186,  114.5944),
    "Mataram":     (-8.5833,  116.1167),
    "Kupang":      (-10.1772, 123.6070),
    "Ambon":       (-3.6954,  128.1814),
    "Jayapura":    (-2.5337,  140.7181),
}


def parse_uploaded_file(file_bytes: bytes, filename: str) -> Tuple[pd.DataFrame, str]:
    """
    Parse uploaded CSV or Excel file into a DataFrame.
    Returns (df, error_message). If error_message is empty, parsing succeeded.
    """
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(file_bytes))
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            return None, f"Format file tidak didukung: {filename}. Gunakan CSV atau Excel (.xlsx/.xls)"

        if df.empty:
            return None, "File kosong. Pastikan file berisi data transaksi."

        # Check required columns
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            return None, f"Kolom wajib tidak ditemukan: {', '.join(missing)}"

        return df, ""

    except Exception as e:
        return None, f"Gagal membaca file: {str(e)}"


def prepare_transactions(df: pd.DataFrame) -> list:
    """
    Convert DataFrame rows to transaction dicts with defaults for missing columns.
    """
    transactions = []
    for i, row in df.iterrows():
        txn = {}
        for col, default in COLUMN_DEFAULTS.items():
            if col in row and pd.notna(row[col]):
                val = row[col]
                # Normalize boolean-like columns
                if col in ["is_new_device", "location_mismatch"]:
                    val = int(bool(val)) if not isinstance(val, bool) else int(val)
                txn[col] = val
            else:
                txn[col] = default(i) if callable(default) else default

        # Ensure transaction_id is string
        txn["transaction_id"] = str(txn["transaction_id"])
        transactions.append(txn)

    return transactions


def run_batch_prediction(transactions: list, predictor_fn) -> pd.DataFrame:
    """
    Run ML prediction on a list of transactions.
    Returns a DataFrame with results.
    """
    results = []
    for txn in transactions:
        try:
            start = time.time()
            ml_result = predictor_fn(txn)
            elapsed = round((time.time() - start) * 1000, 1)

            results.append({
                "transaction_id":    txn.get("transaction_id", ""),
                "merchant_id":       txn.get("merchant_id", ""),
                "location":          txn.get("location", "Jakarta"),
                "amount":            float(txn.get("amount", 0)),
                "hour":              int(txn.get("hour", 12)),
                "day_of_week":       int(txn.get("day_of_week", 1)),
                "is_new_device":     int(txn.get("is_new_device", 0)),
                "location_mismatch": int(txn.get("location_mismatch", 0)),
                "amount_vs_avg_ratio": float(txn.get("amount_vs_avg_ratio", 1.0)),
                "transaction_count_1h": int(txn.get("transaction_count_1h", 1)),
                "lgb_score":         ml_result["lgb_score"],
                "ensemble_score":    ml_result["ensemble_score"],
                "decision":          ml_result["decision"],
                "risk_level":        ml_result["risk_level"],
                "explanation":       ml_result["explanation"],
                "processing_ms":     elapsed,
            })
        except Exception as e:
            results.append({
                "transaction_id":    txn.get("transaction_id", ""),
                "merchant_id":       txn.get("merchant_id", ""),
                "location":          txn.get("location", "Jakarta"),
                "amount":            float(txn.get("amount", 0)),
                "hour":              int(txn.get("hour", 12)),
                "day_of_week":       int(txn.get("day_of_week", 1)),
                "is_new_device":     int(txn.get("is_new_device", 0)),
                "location_mismatch": int(txn.get("location_mismatch", 0)),
                "amount_vs_avg_ratio": float(txn.get("amount_vs_avg_ratio", 1.0)),
                "transaction_count_1h": int(txn.get("transaction_count_1h", 1)),
                "lgb_score":         0.0,
                "ensemble_score":    0.0,
                "decision":          "ERROR",
                "risk_level":        "UNKNOWN",
                "explanation":       str(e),
                "processing_ms":     0.0,
            })

    return pd.DataFrame(results)


def get_city_fraud_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate fraud statistics per city for mapping.
    """
    if results_df.empty:
        return pd.DataFrame()

    # Normalize location to known cities
    def normalize_city(loc):
        loc_str = str(loc).strip().title()
        # Direct match
        if loc_str in CITY_COORDS:
            return loc_str
        # Partial match
        for city in CITY_COORDS:
            if city.lower() in loc_str.lower() or loc_str.lower() in city.lower():
                return city
        return "Jakarta"  # default fallback

    df = results_df.copy()
    df["city"] = df["location"].apply(normalize_city)

    summary = df.groupby("city").agg(
        total_transactions=("transaction_id", "count"),
        total_amount=("amount", "sum"),
        avg_amount=("amount", "mean"),
        blocked=("decision", lambda x: (x == "BLOCK").sum()),
        step_up=("decision", lambda x: (x == "STEP_UP_AUTH").sum()),
        approved=("decision", lambda x: (x == "APPROVE").sum()),
        avg_risk_score=("ensemble_score", "mean"),
        max_risk_score=("ensemble_score", "max"),
    ).reset_index()

    summary["fraud_rate"] = (summary["blocked"] / summary["total_transactions"] * 100).round(1)
    summary["suspicious_rate"] = ((summary["blocked"] + summary["step_up"]) / summary["total_transactions"] * 100).round(1)

    # Add coordinates
    summary["lat"] = summary["city"].map(lambda c: CITY_COORDS.get(c, (-6.2, 106.8))[0])
    summary["lon"] = summary["city"].map(lambda c: CITY_COORDS.get(c, (-6.2, 106.8))[1])

    return summary.sort_values("total_transactions", ascending=False)


def get_temporal_summary(results_df: pd.DataFrame) -> dict:
    """
    Aggregate fraud statistics over time dimensions for tracking charts.
    """
    if results_df.empty:
        return {}

    df = results_df.copy()

    # By hour
    hourly = df.groupby("hour").agg(
        total=("transaction_id", "count"),
        blocked=("decision", lambda x: (x == "BLOCK").sum()),
        avg_score=("ensemble_score", "mean"),
    ).reset_index()
    hourly["fraud_rate"] = (hourly["blocked"] / hourly["total"] * 100).round(1)

    # By day of week
    day_names = {0: "Senin", 1: "Selasa", 2: "Rabu", 3: "Kamis",
                 4: "Jumat", 5: "Sabtu", 6: "Minggu"}
    daily = df.groupby("day_of_week").agg(
        total=("transaction_id", "count"),
        blocked=("decision", lambda x: (x == "BLOCK").sum()),
        avg_score=("ensemble_score", "mean"),
    ).reset_index()
    daily["day_name"] = daily["day_of_week"].map(day_names)
    daily["fraud_rate"] = (daily["blocked"] / daily["total"] * 100).round(1)

    # By risk level distribution
    risk_dist = df["risk_level"].value_counts().to_dict()

    # By decision distribution
    decision_dist = df["decision"].value_counts().to_dict()

    # Score distribution
    score_bins = pd.cut(df["ensemble_score"],
                        bins=[0, 0.2, 0.4, 0.65, 1.0],
                        labels=["Low (0-0.2)", "Medium (0.2-0.4)", "High (0.4-0.65)", "Critical (>0.65)"])
    score_dist = score_bins.value_counts().to_dict()

    # Top risky transactions
    top_risky = df.nlargest(10, "ensemble_score")[
        ["transaction_id", "merchant_id", "location", "amount",
         "hour", "ensemble_score", "decision", "risk_level"]
    ].to_dict("records")

    return {
        "hourly":        hourly.to_dict("records"),
        "daily":         daily.to_dict("records"),
        "risk_dist":     risk_dist,
        "decision_dist": decision_dist,
        "score_dist":    {str(k): v for k, v in score_dist.items()},
        "top_risky":     top_risky,
        "total":         len(df),
        "blocked":       int((df["decision"] == "BLOCK").sum()),
        "step_up":       int((df["decision"] == "STEP_UP_AUTH").sum()),
        "approved":      int((df["decision"] == "APPROVE").sum()),
        "avg_score":     round(df["ensemble_score"].mean(), 4),
        "total_amount":  round(df["amount"].sum(), 2),
        "flagged_amount": round(df[df["decision"] != "APPROVE"]["amount"].sum(), 2),
    }
