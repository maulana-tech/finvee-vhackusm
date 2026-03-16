"""
AEGIS SME — Synthetic Transaction Data Generator
Team Finvee | Varsity Hackathon 2026
Generates realistic SME transaction data with fraud patterns for ASEAN context.
"""

import numpy as np
import pandas as pd
from faker import Faker
import random
from datetime import datetime, timedelta

fake = Faker('id_ID')
np.random.seed(42)
random.seed(42)

MERCHANTS = [
    {"id": "UKM001", "name": "Warung Makan Bu Sari", "type": "food", "city": "Surabaya"},
    {"id": "UKM002", "name": "Toko Batik Pak Hendra", "type": "retail", "city": "Solo"},
    {"id": "UKM003", "name": "Bengkel Motor Jaya", "type": "service", "city": "Bandung"},
    {"id": "UKM004", "name": "Toko Sembako Maju", "type": "grocery", "city": "Medan"},
    {"id": "UKM005", "name": "Salon Kecantikan Ayu", "type": "beauty", "city": "Jakarta"},
]

CITIES = ["Jakarta", "Surabaya", "Bandung", "Medan", "Solo", "Makassar", "Palembang", "Semarang"]
DEVICES = [f"DEV_{i:04d}" for i in range(1, 201)]

def generate_normal_transaction(merchant, timestamp, known_devices):
    """Generate a normal (legitimate) transaction."""
    hour = timestamp.hour
    # Business hours have higher amounts
    if 8 <= hour <= 20:
        amount = np.random.lognormal(mean=12.5, sigma=1.2)  # ~Rp 270k average
    else:
        amount = np.random.lognormal(mean=11.5, sigma=0.8)  # ~Rp 100k average

    return {
        "transaction_id": f"TXN{random.randint(100000, 999999)}",
        "merchant_id": merchant["id"],
        "merchant_type": merchant["type"],
        "timestamp": timestamp,
        "hour": hour,
        "day_of_week": timestamp.weekday(),
        "amount": round(amount, 2),
        "location": merchant["city"],
        "device_id": random.choice(known_devices[:5]),  # Known devices
        "is_new_device": 0,
        "transaction_count_1h": random.randint(1, 8),
        "transaction_count_24h": random.randint(5, 40),
        "amount_vs_avg_ratio": np.random.normal(1.0, 0.3),
        "location_mismatch": 0,
        "is_fraud": 0,
        "fraud_type": "none"
    }

def generate_fraud_transaction(merchant, timestamp, known_devices):
    """Generate a fraudulent transaction with various fraud patterns."""
    fraud_type = random.choice(["account_takeover", "card_fraud", "identity_theft", "unusual_amount"])

    base = generate_normal_transaction(merchant, timestamp, known_devices)

    if fraud_type == "account_takeover":
        base["device_id"] = random.choice(DEVICES[100:])  # Unknown device
        base["is_new_device"] = 1
        base["location"] = random.choice([c for c in CITIES if c != merchant["city"]])
        base["location_mismatch"] = 1
        base["amount"] = round(np.random.uniform(2000000, 15000000), 2)
        base["hour"] = random.choice([1, 2, 3, 4])  # Late night

    elif fraud_type == "card_fraud":
        base["transaction_count_1h"] = random.randint(15, 30)  # Rapid transactions
        base["amount"] = round(np.random.uniform(500000, 3000000), 2)
        base["amount_vs_avg_ratio"] = random.uniform(5, 20)

    elif fraud_type == "identity_theft":
        base["device_id"] = random.choice(DEVICES[150:])
        base["is_new_device"] = 1
        base["location_mismatch"] = 1
        base["amount"] = round(np.random.uniform(1000000, 8000000), 2)

    elif fraud_type == "unusual_amount":
        base["amount"] = round(np.random.uniform(10000000, 50000000), 2)
        base["amount_vs_avg_ratio"] = random.uniform(15, 50)

    base["is_fraud"] = 1
    base["fraud_type"] = fraud_type
    return base

def generate_dataset(n_transactions=10000, fraud_ratio=0.025):
    """Generate full dataset with normal and fraud transactions."""
    records = []
    start_date = datetime(2025, 1, 1)

    for merchant in MERCHANTS:
        known_devices = random.sample(DEVICES[:50], 10)
        n_merchant = n_transactions // len(MERCHANTS)
        n_fraud = int(n_merchant * fraud_ratio)
        n_normal = n_merchant - n_fraud

        for i in range(n_normal):
            ts = start_date + timedelta(
                days=random.randint(0, 365),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            records.append(generate_normal_transaction(merchant, ts, known_devices))

        for i in range(n_fraud):
            ts = start_date + timedelta(
                days=random.randint(0, 365),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            records.append(generate_fraud_transaction(merchant, ts, known_devices))

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

if __name__ == "__main__":
    print("Generating synthetic transaction dataset...")
    df = generate_dataset(n_transactions=10000, fraud_ratio=0.025)
    df.to_csv("/home/ubuntu/aegis-sme/data/transactions.csv", index=False)
    print(f"Dataset generated: {len(df)} transactions")
    print(f"Fraud rate: {df['is_fraud'].mean():.2%}")
    print(f"Columns: {list(df.columns)}")
    print(df['fraud_type'].value_counts())
