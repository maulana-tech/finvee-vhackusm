"""
AEGIS SME — FastAPI Backend
Team Finvee | Varsity Hackathon 2026

REST API that integrates ML Core Engine + Multi-Agent System.
Endpoints:
  POST /analyze          — Analyze a single transaction
  POST /batch-analyze    — Analyze multiple transactions
  GET  /cases            — Get all processed cases
  GET  /notifications    — Get all notifications
  GET  /stats            — Get system statistics
  GET  /health           — Health check
  POST /demo/simulate    — Simulate a transaction stream for demo
"""

import sys
import os
import random
import json
import time
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, "/home/ubuntu/aegis-sme")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from ml.predictor import predict, get_model_info
from agents.aegis_agents import AegisOrchestrator, CASE_LOG, NOTIFICATION_LOG, TRANSACTION_HISTORY

app = FastAPI(
    title="AEGIS SME API",
    description="Autonomous Financial Guardian for SMEs — Team Finvee, Varsity Hackathon 2026",
    version="2.0.0-ieee"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = AegisOrchestrator()

# ─── Pydantic Models ───
class Transaction(BaseModel):
    transaction_id: str
    merchant_id: str
    merchant_type: str = "food"
    amount: float
    hour: int
    day_of_week: int = 1
    location: str
    device_id: str = "DEV_0001"
    is_new_device: int = 0
    transaction_count_1h: int = 3
    transaction_count_24h: int = 15
    amount_vs_avg_ratio: float = 1.0
    location_mismatch: int = 0

class BatchRequest(BaseModel):
    transactions: list[Transaction]

# ─── Endpoints ───

@app.get("/health")
def health_check():
    try:
        info = get_model_info()
        model_status = "loaded"
    except Exception:
        info = {}
        model_status = "not_loaded"
    return {
        "status": "healthy",
        "service": "AEGIS SME",
        "team": "Finvee",
        "version": "2.0.0-ieee",
        "model_version": info.get("model_version", "IEEE-CIS v1"),
        "model_status": model_status,
        "lgb_auc": info.get("lgb_auc", "N/A"),
        "ensemble_auc": info.get("ensemble_auc", "N/A"),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model-info")
def model_info():
    """Get detailed model metadata and performance metrics."""
    try:
        return get_model_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
def analyze_transaction(txn: Transaction):
    """Analyze a single transaction through full ML + Agent pipeline."""
    txn_dict = txn.model_dump()
    try:
        ml_result = predict(txn_dict)
        case = orchestrator.process(txn_dict, ml_result)
        return {
            "success": True,
            "case_id": case["case_id"],
            "transaction_id": txn.transaction_id,
            "final_action": case["final_action"],
            "risk_score": ml_result["ensemble_score"],
            "risk_level": ml_result["risk_level"],
            "explanation": ml_result["explanation"],
            "reasoning": case["resolution"]["reasoning"],
            "notification": case["notification"]["message"],
            "evidence_flags": case["investigation"]["evidence_flags"],
            "fraud_confidence": case["investigation"]["fraud_confidence"],
            "processing_time_ms": case["processing_time_ms"],
            "agents_involved": [
                case["monitor_result"]["agent"],
                case["investigation"]["agent"],
                case["resolution"]["agent"],
                case["notification"]["agent"]
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-analyze")
def batch_analyze(req: BatchRequest):
    """Analyze multiple transactions."""
    results = []
    for txn in req.transactions:
        txn_dict = txn.model_dump()
        ml_result = predict(txn_dict)
        case = orchestrator.process(txn_dict, ml_result)
        results.append({
            "transaction_id": txn.transaction_id,
            "final_action": case["final_action"],
            "risk_score": ml_result["ensemble_score"],
            "risk_level": ml_result["risk_level"],
            "processing_time_ms": case["processing_time_ms"]
        })
    return {"success": True, "count": len(results), "results": results}

@app.get("/cases")
def get_cases(limit: int = 50):
    """Get recent processed cases."""
    recent = CASE_LOG[-limit:]
    return {
        "total": len(CASE_LOG),
        "returned": len(recent),
        "cases": [
            {
                "case_id": c["case_id"],
                "transaction_id": c["transaction"]["transaction_id"],
                "merchant_id": c["transaction"]["merchant_id"],
                "amount": c["transaction"]["amount"],
                "final_action": c["final_action"],
                "risk_score": c["ml_result"]["ensemble_score"],
                "evidence_flags": c["investigation"]["evidence_flags"],
                "timestamp": c["resolution"]["timestamp"],
                "processing_time_ms": c["processing_time_ms"]
            }
            for c in reversed(recent)
        ]
    }

@app.get("/notifications")
def get_notifications(limit: int = 20):
    """Get recent notifications sent to SME owners."""
    recent = NOTIFICATION_LOG[-limit:]
    return {
        "total": len(NOTIFICATION_LOG),
        "returned": len(recent),
        "notifications": list(reversed(recent))
    }

@app.get("/stats")
def get_stats():
    """Get system-wide statistics."""
    if not CASE_LOG:
        return {"message": "No transactions processed yet"}

    total = len(CASE_LOG)
    actions = [c["final_action"] for c in CASE_LOG]
    blocked = actions.count("BLOCK")
    step_up = actions.count("STEP_UP_AUTH")
    approved = actions.count("APPROVE")
    avg_time = sum(c["processing_time_ms"] for c in CASE_LOG) / total

    scores = [c["ml_result"]["ensemble_score"] for c in CASE_LOG]
    high_risk = sum(1 for s in scores if s >= 0.75)

    return {
        "total_transactions": total,
        "blocked": blocked,
        "step_up_auth": step_up,
        "approved": approved,
        "block_rate": round(blocked / total * 100, 2),
        "avg_risk_score": round(sum(scores) / len(scores), 4),
        "high_risk_transactions": high_risk,
        "avg_processing_time_ms": round(avg_time, 1),
        "total_notifications": len(NOTIFICATION_LOG)
    }

@app.post("/demo/simulate")
def simulate_stream(n: int = 10):
    """Simulate a stream of transactions for demo purposes."""
    MERCHANTS = ["UKM001", "UKM002", "UKM003", "UKM004", "UKM005"]
    MERCHANT_TYPES = ["food", "retail", "service", "grocery", "beauty"]
    CITIES = ["Jakarta", "Surabaya", "Bandung", "Medan", "Solo"]

    results = []
    for i in range(min(n, 20)):
        is_fraud = random.random() < 0.25
        merchant_idx = random.randint(0, 4)

        if is_fraud:
            txn_dict = {
                "transaction_id": f"TXN_DEMO_{int(time.time()*1000)}_{i}",
                "merchant_id": MERCHANTS[merchant_idx],
                "merchant_type": MERCHANT_TYPES[merchant_idx],
                "amount": round(random.uniform(3000000, 15000000), 2),
                "hour": random.choice([1, 2, 3, 23]),
                "day_of_week": random.randint(0, 6),
                "location": random.choice([c for c in CITIES if c != CITIES[merchant_idx]]),
                "device_id": f"DEV_{random.randint(500, 999):04d}",
                "is_new_device": 1,
                "transaction_count_1h": random.randint(1, 3),
                "transaction_count_24h": random.randint(3, 10),
                "amount_vs_avg_ratio": round(random.uniform(8, 20), 2),
                "location_mismatch": 1
            }
        else:
            txn_dict = {
                "transaction_id": f"TXN_DEMO_{int(time.time()*1000)}_{i}",
                "merchant_id": MERCHANTS[merchant_idx],
                "merchant_type": MERCHANT_TYPES[merchant_idx],
                "amount": round(random.uniform(50000, 800000), 2),
                "hour": random.randint(8, 20),
                "day_of_week": random.randint(0, 6),
                "location": CITIES[merchant_idx],
                "device_id": f"DEV_{random.randint(1, 10):04d}",
                "is_new_device": 0,
                "transaction_count_1h": random.randint(1, 5),
                "transaction_count_24h": random.randint(5, 30),
                "amount_vs_avg_ratio": round(random.uniform(0.5, 2.5), 2),
                "location_mismatch": 0
            }

        ml_result = predict(txn_dict)
        case = orchestrator.process(txn_dict, ml_result)
        results.append({
            "transaction_id": txn_dict["transaction_id"],
            "merchant_id": txn_dict["merchant_id"],
            "amount": txn_dict["amount"],
            "final_action": case["final_action"],
            "risk_score": ml_result["ensemble_score"],
            "risk_level": ml_result["risk_level"],
            "notification": case["notification"]["message"][:100] + "..."
        })

    return {
        "success": True,
        "simulated": len(results),
        "results": results,
        "summary": {
            "blocked": sum(1 for r in results if r["final_action"] == "BLOCK"),
            "step_up": sum(1 for r in results if r["final_action"] == "STEP_UP_AUTH"),
            "approved": sum(1 for r in results if r["final_action"] == "APPROVE"),
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
