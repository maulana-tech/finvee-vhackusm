"""
AEGIS SME — Multi-Agent System
Team Finvee | Varsity Hackathon 2026

4 Agents:
  1. Monitor Agent       — Watches transaction stream, triggers on threshold
  2. Investigator Agent  — Gathers evidence via tool calls
  3. Resolution Agent    — Decides final action
  4. Communicator Agent  — Notifies SME owner in local language

Orchestrated by: Orchestrator Agent
"""

import os
import json
import time
import random
from datetime import datetime
from typing import Any
from openai import OpenAI

# ─── OpenAI Client (uses env var OPENAI_API_KEY + base_url) ───
client = OpenAI()

# ─── In-memory stores (simulated databases) ───
TRANSACTION_HISTORY = {}   # merchant_id -> list of past transactions
FRAUD_PATTERN_DB = [
    "Transaksi dari perangkat baru + lokasi berbeda + jumlah besar = Account Takeover",
    "Banyak transaksi kecil dalam waktu singkat = Card Testing Fraud",
    "Jumlah transaksi > 10x rata-rata = Unusual Amount Fraud",
    "Lokasi berbeda dalam waktu < 30 menit = Impossible Travel",
    "Perangkat baru + jam dini hari + jumlah besar = Identity Theft",
]
NOTIFICATION_LOG = []      # All notifications sent
CASE_LOG = []              # Full investigation case log

# ─────────────────────────────────────────────────────────────
# TOOL DEFINITIONS (called by Investigator Agent)
# ─────────────────────────────────────────────────────────────

def get_user_history(merchant_id: str, limit: int = 10) -> dict:
    """Retrieve recent transaction history for a merchant."""
    history = TRANSACTION_HISTORY.get(merchant_id, [])
    recent = history[-limit:] if len(history) >= limit else history
    if not recent:
        return {"merchant_id": merchant_id, "history": [], "avg_amount": 0, "avg_count_per_day": 0}
    amounts = [t["amount"] for t in recent]
    return {
        "merchant_id": merchant_id,
        "total_transactions": len(history),
        "recent_transactions": recent,
        "avg_amount": round(sum(amounts) / len(amounts), 2),
        "max_amount": max(amounts),
        "min_amount": min(amounts),
        "avg_count_per_day": round(len(history) / 30, 1)
    }

def check_location_consistency(current_location: str, merchant_id: str) -> dict:
    """Check if current location matches merchant's historical locations."""
    history = TRANSACTION_HISTORY.get(merchant_id, [])
    if not history:
        return {"consistent": True, "historical_locations": [], "current_location": current_location}
    historical_locs = list(set(t.get("location", "") for t in history))
    consistent = current_location in historical_locs
    return {
        "consistent": consistent,
        "current_location": current_location,
        "historical_locations": historical_locs,
        "mismatch": not consistent
    }

def analyze_device_fingerprint(device_id: str, merchant_id: str) -> dict:
    """Check if device is known for this merchant."""
    history = TRANSACTION_HISTORY.get(merchant_id, [])
    known_devices = list(set(t.get("device_id", "") for t in history))
    is_known = device_id in known_devices
    return {
        "device_id": device_id,
        "is_known_device": is_known,
        "known_devices_count": len(known_devices),
        "risk_note": "Perangkat tidak dikenal — risiko tinggi" if not is_known else "Perangkat dikenal — aman"
    }

def query_fraud_pattern_db(features: dict) -> dict:
    """Match transaction features against known fraud patterns."""
    matched_patterns = []
    risk_score_boost = 0.0

    if features.get("is_new_device") and features.get("location_mismatch"):
        matched_patterns.append(FRAUD_PATTERN_DB[0])
        risk_score_boost += 0.3
    if features.get("transaction_count_1h", 0) > 10:
        matched_patterns.append(FRAUD_PATTERN_DB[1])
        risk_score_boost += 0.2
    if features.get("amount_vs_avg_ratio", 1) > 8:
        matched_patterns.append(FRAUD_PATTERN_DB[2])
        risk_score_boost += 0.25
    if features.get("hour", 12) in [0, 1, 2, 3] and features.get("is_new_device"):
        matched_patterns.append(FRAUD_PATTERN_DB[4])
        risk_score_boost += 0.2

    return {
        "matched_patterns": matched_patterns,
        "pattern_count": len(matched_patterns),
        "risk_score_boost": round(risk_score_boost, 2),
        "verdict": "SUSPICIOUS" if matched_patterns else "CLEAN"
    }

def get_merchant_profile(merchant_id: str) -> dict:
    """Get merchant profile information."""
    profiles = {
        "UKM001": {"name": "Warung Makan Bu Sari", "type": "food", "city": "Surabaya", "owner": "Bu Sari"},
        "UKM002": {"name": "Toko Batik Pak Hendra", "type": "retail", "city": "Solo", "owner": "Pak Hendra"},
        "UKM003": {"name": "Bengkel Motor Jaya", "type": "service", "city": "Bandung", "owner": "Pak Jaya"},
        "UKM004": {"name": "Toko Sembako Maju", "type": "grocery", "city": "Medan", "owner": "Ibu Maju"},
        "UKM005": {"name": "Salon Kecantikan Ayu", "type": "beauty", "city": "Jakarta", "owner": "Ibu Ayu"},
    }
    return profiles.get(merchant_id, {"name": "Unknown", "type": "unknown", "city": "Unknown", "owner": "Pemilik"})

# ─────────────────────────────────────────────────────────────
# AGENT IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────────

class MonitorAgent:
    """Agent 1: Watches transaction stream and triggers investigation."""

    def __init__(self, ml_threshold_flag=0.45, ml_threshold_block=0.75):
        self.threshold_flag = ml_threshold_flag
        self.threshold_block = ml_threshold_block

    def evaluate(self, transaction: dict, ml_result: dict) -> dict:
        score = ml_result["ensemble_score"]
        decision = ml_result["decision"]

        action = "PASS"
        trigger_reason = ""

        if score >= self.threshold_block:
            action = "TRIGGER_BLOCK"
            trigger_reason = f"Skor risiko sangat tinggi ({score:.2f}). Langsung blokir."
        elif score >= self.threshold_flag:
            action = "TRIGGER_INVESTIGATE"
            trigger_reason = f"Skor risiko mencurigakan ({score:.2f}). Perlu investigasi."

        return {
            "agent": "MonitorAgent",
            "action": action,
            "trigger_reason": trigger_reason,
            "ml_score": score,
            "ml_decision": decision,
            "timestamp": datetime.now().isoformat()
        }


class InvestigatorAgent:
    """Agent 2: Gathers evidence using tool calls."""

    def investigate(self, transaction: dict, ml_result: dict) -> dict:
        merchant_id = transaction.get("merchant_id", "UKM001")
        device_id = transaction.get("device_id", "DEV_UNKNOWN")
        location = transaction.get("location", "Unknown")

        # Run all investigation tools
        history = get_user_history(merchant_id)
        location_check = check_location_consistency(location, merchant_id)
        device_check = analyze_device_fingerprint(device_id, merchant_id)
        pattern_match = query_fraud_pattern_db({
            "is_new_device": transaction.get("is_new_device", 0),
            "location_mismatch": transaction.get("location_mismatch", 0),
            "transaction_count_1h": transaction.get("transaction_count_1h", 1),
            "amount_vs_avg_ratio": transaction.get("amount_vs_avg_ratio", 1.0),
            "hour": transaction.get("hour", 12)
        })
        merchant_profile = get_merchant_profile(merchant_id)

        # Build evidence summary
        evidence_flags = []
        if not device_check["is_known_device"]:
            evidence_flags.append("PERANGKAT_BARU")
        if location_check["mismatch"]:
            evidence_flags.append("LOKASI_MISMATCH")
        if pattern_match["pattern_count"] > 0:
            evidence_flags.append(f"POLA_FRAUD_TERDETEKSI({pattern_match['pattern_count']})")
        if transaction.get("amount_vs_avg_ratio", 1) > 5:
            evidence_flags.append("JUMLAH_TIDAK_WAJAR")

        confidence = min(0.5 + (len(evidence_flags) * 0.15) + ml_result["ensemble_score"] * 0.3, 1.0)

        return {
            "agent": "InvestigatorAgent",
            "merchant_profile": merchant_profile,
            "history_summary": {
                "total_transactions": history["total_transactions"],
                "avg_amount": history["avg_amount"],
                "max_amount": history["max_amount"]
            },
            "location_check": location_check,
            "device_check": device_check,
            "pattern_match": pattern_match,
            "evidence_flags": evidence_flags,
            "fraud_confidence": round(confidence, 2),
            "tools_called": ["get_user_history", "check_location_consistency",
                             "analyze_device_fingerprint", "query_fraud_pattern_db",
                             "get_merchant_profile"]
        }


class ResolutionAgent:
    """Agent 3: Makes final decision based on investigation report."""

    def resolve(self, transaction: dict, ml_result: dict,
                monitor_result: dict, investigation: dict) -> dict:
        confidence = investigation["fraud_confidence"]
        flags = investigation["evidence_flags"]
        score = ml_result["ensemble_score"]
        monitor_action = monitor_result["action"]

        # Decision logic
        if monitor_action == "TRIGGER_BLOCK" or confidence >= 0.80:
            final_action = "BLOCK"
            action_detail = "Transaksi diblokir otomatis. Terlalu banyak indikator fraud."
            notification_type = "FRAUD_BLOCKED"
        elif confidence >= 0.55 or len(flags) >= 2:
            final_action = "STEP_UP_AUTH"
            action_detail = "Verifikasi tambahan diperlukan sebelum transaksi diproses."
            notification_type = "STEP_UP_REQUIRED"
        else:
            final_action = "APPROVE"
            action_detail = "Transaksi disetujui. Profil risiko dalam batas normal."
            notification_type = "APPROVED"

        # Generate LLM-powered reasoning
        reasoning = self._generate_reasoning(transaction, ml_result, investigation, final_action)

        return {
            "agent": "ResolutionAgent",
            "final_action": final_action,
            "action_detail": action_detail,
            "notification_type": notification_type,
            "fraud_confidence": confidence,
            "evidence_flags": flags,
            "reasoning": reasoning,
            "timestamp": datetime.now().isoformat()
        }

    def _generate_reasoning(self, transaction, ml_result, investigation, action):
        """Use LLM to generate human-readable reasoning."""
        prompt = f"""Kamu adalah Resolution Agent dari sistem AEGIS SME — sistem keamanan finansial untuk UKM.

Berdasarkan data berikut, berikan reasoning singkat (2-3 kalimat) mengapa keputusan "{action}" diambil:

Transaksi:
- ID: {transaction.get('transaction_id')}
- Merchant: {investigation['merchant_profile']['name']}
- Jumlah: Rp {transaction.get('amount', 0):,.0f}
- Lokasi: {transaction.get('location')}
- Jam: {transaction.get('hour', 0):02d}:00

Hasil ML:
- Ensemble Risk Score: {ml_result['ensemble_score']:.2f}
- Faktor: {ml_result['explanation']}

Bukti Investigasi:
- Flag: {', '.join(investigation['evidence_flags']) if investigation['evidence_flags'] else 'Tidak ada'}
- Pola fraud cocok: {investigation['pattern_match']['pattern_count']} pola
- Fraud confidence: {investigation['fraud_confidence']:.0%}

Berikan reasoning singkat dalam Bahasa Indonesia yang profesional."""

        try:
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Keputusan {action} diambil berdasarkan analisis risiko ensemble dengan confidence {investigation['fraud_confidence']:.0%}."


class CommunicatorAgent:
    """Agent 4: Notifies SME owner in local language."""

    def notify(self, transaction: dict, resolution: dict, investigation: dict) -> dict:
        merchant_profile = investigation["merchant_profile"]
        owner_name = merchant_profile.get("owner", "Pemilik")
        action = resolution["final_action"]
        amount = transaction.get("amount", 0)
        location = transaction.get("location", "")
        txn_id = transaction.get("transaction_id", "")

        # Generate notification message
        message = self._generate_notification(
            owner_name, action, amount, location, txn_id,
            resolution, investigation
        )

        # Generate action buttons based on decision
        if action == "BLOCK":
            buttons = ["Lihat Detail", "Hubungi Support", "Buka Blokir (Verifikasi)"]
        elif action == "STEP_UP_AUTH":
            buttons = ["Ya, Ini Saya", "Bukan Saya - Blokir", "Tanya CS"]
        else:
            buttons = ["Lihat Detail", "OK"]

        notification = {
            "agent": "CommunicatorAgent",
            "recipient": owner_name,
            "merchant_id": transaction.get("merchant_id"),
            "notification_type": resolution["notification_type"],
            "message": message,
            "action_buttons": buttons,
            "priority": "HIGH" if action == "BLOCK" else ("MEDIUM" if action == "STEP_UP_AUTH" else "LOW"),
            "channel": "WhatsApp-style UI",
            "timestamp": datetime.now().isoformat(),
            "transaction_id": txn_id
        }

        NOTIFICATION_LOG.append(notification)
        return notification

    def _generate_notification(self, owner_name, action, amount, location,
                                txn_id, resolution, investigation):
        """Generate localized notification message using LLM."""
        action_context = {
            "BLOCK": "transaksi ini DIBLOKIR karena terdeteksi sebagai fraud",
            "STEP_UP_AUTH": "transaksi ini DITAHAN sementara dan memerlukan konfirmasi Anda",
            "APPROVE": "transaksi ini DISETUJUI dan sudah diproses"
        }

        prompt = f"""Kamu adalah Communicator Agent dari AEGIS SME. Tulis notifikasi WhatsApp yang singkat, jelas, dan ramah untuk pemilik UKM.

Data:
- Nama pemilik: {owner_name}
- Aksi: {action_context.get(action, action)}
- Jumlah: Rp {amount:,.0f}
- Lokasi transaksi: {location}
- ID Transaksi: {txn_id}
- Alasan: {resolution['reasoning'][:100]}...

Syarat notifikasi:
1. Maksimal 4 kalimat
2. Bahasa Indonesia yang ramah dan mudah dipahami
3. Sebutkan jumlah dan lokasi transaksi
4. Jika BLOCK atau STEP_UP: minta konfirmasi dengan jelas
5. Gunakan format pesan WhatsApp (tidak perlu salam panjang)"""

        try:
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=120,
                temperature=0.4
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if action == "BLOCK":
                return f"⚠️ {owner_name}, transaksi Rp {amount:,.0f} dari {location} DIBLOKIR karena terdeteksi mencurigakan. ID: {txn_id}. Hubungi support jika ini transaksi Anda."
            elif action == "STEP_UP_AUTH":
                return f"🔔 {owner_name}, ada transaksi Rp {amount:,.0f} dari {location} yang perlu konfirmasi Anda. ID: {txn_id}. Apakah ini transaksi Anda?"
            else:
                return f"✅ {owner_name}, transaksi Rp {amount:,.0f} dari {location} berhasil diproses. ID: {txn_id}."


# ─────────────────────────────────────────────────────────────
# ORCHESTRATOR — Coordinates all agents
# ─────────────────────────────────────────────────────────────

class AegisOrchestrator:
    """
    Main orchestrator that coordinates all 4 agents.
    Receives ML result and transaction, runs full agentic workflow.
    """

    def __init__(self):
        self.monitor = MonitorAgent()
        self.investigator = InvestigatorAgent()
        self.resolution = ResolutionAgent()
        self.communicator = CommunicatorAgent()

    def process(self, transaction: dict, ml_result: dict) -> dict:
        """Run full agentic pipeline for a transaction."""
        start_time = time.time()
        case_id = f"CASE-{transaction.get('transaction_id', 'UNKNOWN')}"

        # Add to history
        merchant_id = transaction.get("merchant_id", "UKM001")
        if merchant_id not in TRANSACTION_HISTORY:
            TRANSACTION_HISTORY[merchant_id] = []
        TRANSACTION_HISTORY[merchant_id].append(transaction)

        # Step 1: Monitor Agent
        monitor_result = self.monitor.evaluate(transaction, ml_result)

        # Step 2: Investigator Agent (only if flagged)
        if monitor_result["action"] in ["TRIGGER_INVESTIGATE", "TRIGGER_BLOCK"]:
            investigation = self.investigator.investigate(transaction, ml_result)
        else:
            # Low risk — minimal investigation
            investigation = {
                "agent": "InvestigatorAgent",
                "merchant_profile": get_merchant_profile(merchant_id),
                "history_summary": {"total_transactions": len(TRANSACTION_HISTORY.get(merchant_id, [])), "avg_amount": 0, "max_amount": 0},
                "location_check": {"consistent": True, "mismatch": False},
                "device_check": {"is_known_device": True},
                "pattern_match": {"matched_patterns": [], "pattern_count": 0, "risk_score_boost": 0},
                "evidence_flags": [],
                "fraud_confidence": ml_result["ensemble_score"] * 0.5,
                "tools_called": ["get_merchant_profile"]
            }

        # Step 3: Resolution Agent
        resolution = self.resolution.resolve(transaction, ml_result, monitor_result, investigation)

        # Step 4: Communicator Agent
        notification = self.communicator.notify(transaction, resolution, investigation)

        elapsed = round((time.time() - start_time) * 1000, 1)

        # Build full case record
        case = {
            "case_id": case_id,
            "transaction": transaction,
            "ml_result": ml_result,
            "monitor_result": monitor_result,
            "investigation": investigation,
            "resolution": resolution,
            "notification": notification,
            "processing_time_ms": elapsed,
            "final_action": resolution["final_action"]
        }

        CASE_LOG.append(case)
        return case


# ─────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/ubuntu/aegis-sme")
    from ml.predictor import predict

    print("=" * 60)
    print("  AEGIS SME — Multi-Agent System Test")
    print("=" * 60)

    test_txn = {
        "transaction_id": "TXN_TEST_001",
        "merchant_id": "UKM001",
        "merchant_type": "food",
        "amount": 9500000,
        "hour": 2,
        "day_of_week": 1,
        "location": "Jakarta",
        "device_id": "DEV_9999",
        "is_new_device": 1,
        "transaction_count_1h": 1,
        "transaction_count_24h": 5,
        "amount_vs_avg_ratio": 14.5,
        "location_mismatch": 1
    }

    # Seed some history
    for i in range(20):
        TRANSACTION_HISTORY.setdefault("UKM001", []).append({
            "amount": random.uniform(200000, 800000),
            "location": "Surabaya",
            "device_id": f"DEV_{random.randint(1,5):04d}",
            "hour": random.randint(8, 20)
        })

    print("\n[TEST] Running ML prediction...")
    ml_result = predict(test_txn)
    print(f"  Ensemble Score: {ml_result['ensemble_score']:.4f}")
    print(f"  Decision: {ml_result['decision']}")

    print("\n[TEST] Running Agentic Pipeline...")
    orchestrator = AegisOrchestrator()
    case = orchestrator.process(test_txn, ml_result)

    print(f"\n{'='*60}")
    print(f"  CASE ID: {case['case_id']}")
    print(f"  FINAL ACTION: {case['final_action']}")
    print(f"  Processing Time: {case['processing_time_ms']}ms")
    print(f"\n  Monitor: {case['monitor_result']['action']}")
    print(f"  Evidence Flags: {case['investigation']['evidence_flags']}")
    print(f"  Fraud Confidence: {case['investigation']['fraud_confidence']:.0%}")
    print(f"\n  Reasoning:\n  {case['resolution']['reasoning']}")
    print(f"\n  Notification to {case['notification']['recipient']}:")
    print(f"  {case['notification']['message']}")
    print("=" * 60)
