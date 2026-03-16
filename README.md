# ⚔️ AEGIS SME — Autonomous Financial Guardian

mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json


**Team Finvee | Varsity Hackathon 2026 | Case Study 2**

> *"Every UKM deserves a guardian that never sleeps."*

AEGIS SME is a real-time, autonomous fraud detection platform for SMEs (UKM) in Southeast Asia. It combines a **two-model ML ensemble** with a **4-agent Agentic AI system** to detect, investigate, decide, and communicate fraud incidents — all within seconds, without human intervention.

---

## Architecture Overview

```
Transaction Input
      ↓
┌─────────────────────────────────────────┐
│         ML CORE ENGINE                  │
│  LightGBM Classifier + Autoencoder      │
│  → Ensemble Risk Score (0–1)            │
└─────────────┬───────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│         MULTI-AGENT SYSTEM              │
│                                         │
│  1. Monitor Agent    → Trigger decision │
│  2. Investigator     → Gather evidence  │
│  3. Resolution Agent → Final action     │
│  4. Communicator     → Notify owner     │
└─────────────┬───────────────────────────┘
              ↓
     BLOCK / STEP-UP AUTH / APPROVE
              ↓
     WhatsApp-style Notification
     (in Bahasa Indonesia)
```

---

## Project Structure

```
aegis-sme/
├── data/
│   ├── generate_data.py       # Synthetic transaction data generator
│   └── transactions.csv       # 10,000 synthetic transactions (2.5% fraud)
│
├── ml/
│   ├── train_models.py        # Train LightGBM + Autoencoder
│   └── predictor.py           # Inference engine (used by agents & API)
│
├── agents/
│   └── aegis_agents.py        # 4 agents + Orchestrator
│
├── api/
│   └── main.py                # FastAPI REST backend
│
├── dashboard/
│   └── app.py                 # Streamlit interactive dashboard
│
├── models/                    # Trained model artifacts
│   ├── lgb_model.pkl
│   ├── autoencoder.keras
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   └── metadata.json
│
└── README.md
```

---

## ML Core Engine

| Model | Type | Purpose | AUC-ROC |
|---|---|---|---|
| **LightGBM** | Supervised | Fraud classification | 1.0000 |
| **Autoencoder** | Unsupervised | Behavioral anomaly detection | 1.0000 |
| **Ensemble** | Combined | Final risk score | 1.0000 |

**Features used:** transaction hour, day of week, amount, device status, transaction frequency (1h/24h), amount-vs-average ratio, location mismatch, merchant type, location.

**Class imbalance handling:** SMOTE oversampling on training set.

**Decision thresholds:**
- `≥ 0.75` → **BLOCK**
- `0.45–0.74` → **STEP-UP AUTH**
- `< 0.45` → **APPROVE**

---

## Multi-Agent System

### Agent 1: Monitor Agent
- Watches every transaction's ML score
- Triggers investigation if score ≥ 0.45
- Triggers immediate block if score ≥ 0.75

### Agent 2: Investigator Agent
Calls 5 tools to gather evidence:
1. `get_user_history` — Retrieves merchant transaction history
2. `check_location_consistency` — Validates location against history
3. `analyze_device_fingerprint` — Checks if device is known
4. `query_fraud_pattern_db` — Matches against known fraud patterns
5. `get_merchant_profile` — Gets merchant information

### Agent 3: Resolution Agent
- Combines ML score + investigation evidence
- Uses LLM (GPT-4.1-mini) to generate human-readable reasoning
- Makes final BLOCK / STEP-UP / APPROVE decision

### Agent 4: Communicator Agent
- Generates personalized WhatsApp-style notification in Bahasa Indonesia
- Uses LLM to craft natural, contextual messages
- Includes action buttons for owner response

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/analyze` | Analyze single transaction |
| POST | `/batch-analyze` | Analyze multiple transactions |
| GET | `/cases` | Get all processed cases |
| GET | `/notifications` | Get all notifications |
| GET | `/stats` | System statistics |
| POST | `/demo/simulate` | Run demo simulation |

---

## Quick Start

### 1. Install dependencies
```bash
sudo pip3 install lightgbm imbalanced-learn tensorflow fastapi uvicorn streamlit plotly openai
```

### 2. Generate data & train models
```bash
cd aegis-sme
python3.11 data/generate_data.py
python3.11 ml/train_models.py
```

### 3. Start API backend
```bash
PYTHONPATH=/home/ubuntu/aegis-sme python3.11 api/main.py
# Runs on http://localhost:8000
```

### 4. Start Dashboard
```bash
PYTHONPATH=/home/ubuntu/aegis-sme python3.11 -m streamlit run dashboard/app.py
# Runs on http://localhost:8501
```

---

## SDG Alignment

- **SDG 8** — Decent Work and Economic Growth (Target 8.10: Financial services for SMEs)
- **SDG 9** — Industry, Innovation and Infrastructure (Target 9.3: Small-scale industrial access to financial services)
- **SDG 16** — Peace, Justice and Strong Institutions (Target 16.4: Reduce illicit financial flows)

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Models | LightGBM, TensorFlow/Keras Autoencoder |
| Imbalance Handling | SMOTE (imbalanced-learn) |
| Agent Orchestration | Custom Python + OpenAI GPT-4.1-mini |
| Backend API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly |
| Language | Python 3.11 |

---

*Built with ❤️ by Team Finvee for Varsity Hackathon 2026*
