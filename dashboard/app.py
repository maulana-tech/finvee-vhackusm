"""
AEGIS SME — Dashboard v2.0
Team Finvee | Varsity Hackathon 2026 | Case Study 2

Tabs:
  1. Analyze Transaction   — Single transaction analysis
  2. Batch Upload          — CSV/Excel upload & bulk analysis
  3. Fraud Map             — Geographic fraud mapping per city
  4. Temporal Tracking     — Time-based fraud trends & heatmaps
  5. Live Dashboard        — Real-time stats & charts
  6. Agent Workflow        — Agent reasoning chain
  7. Notifications         — SME owner alerts
"""

import sys
import os
import time
import random
import json
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
from ml.predictor import predict
from agents.aegis_agents import AegisOrchestrator

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AEGIS SME | Finvee",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #0f1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252840);
        border-radius: 12px; padding: 20px;
        border-left: 4px solid #4A90D9; margin-bottom: 10px;
    }
    .badge-block   { background:#E74C3C; color:white; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:600; }
    .badge-stepup  { background:#E67E22; color:white; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:600; }
    .badge-approve { background:#27AE60; color:white; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:600; }
    .notification-card { background:#1a1d2e; border-radius:10px; padding:14px; margin:8px 0; border:1px solid #2a2d3e; }
    .notification-card.HIGH   { border-color:#E74C3C; }
    .notification-card.MEDIUM { border-color:#E67E22; }
    .notification-card.LOW    { border-color:#27AE60; }
    .stButton > button {
        background: linear-gradient(135deg, #4A90D9, #8E44AD);
        color:white; border:none; border-radius:8px;
        padding:10px 24px; font-weight:600; transition:all 0.3s;
    }
    .stButton > button:hover { opacity:0.9; transform:translateY(-1px); }
    div[data-testid="stSidebarContent"] { background:#0d1117; }
    .stTabs [data-baseweb="tab"] { color:#aaa; }
    .stTabs [aria-selected="true"] { color:#4A90D9 !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Constants ─────────────────────────────────────────────────────────────────
PLOTLY_DARK = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#1a1d2e",
    font=dict(color="#ccc", family="Inter"),
)
CITY_COORDS = {
    "Jakarta": (-6.2088, 106.8456),
    "Surabaya": (-7.2575, 112.7521),
    "Bandung": (-6.9175, 107.6191),
    "Medan": (3.5952, 98.6722),
    "Solo": (-7.5755, 110.8243),
    "Semarang": (-6.9932, 110.4203),
    "Yogyakarta": (-7.7956, 110.3695),
    "Makassar": (-5.1477, 119.4327),
    "Palembang": (-2.9761, 104.7754),
    "Denpasar": (-8.6705, 115.2126),
    "Malang": (-7.9797, 112.6304),
    "Pekanbaru": (0.5071, 101.4478),
    "Balikpapan": (-1.2675, 116.8289),
    "Pontianak": (-0.0263, 109.3425),
    "Manado": (1.4748, 124.8421),
    "Padang": (-0.9471, 100.4172),
    "Banjarmasin": (-3.3186, 114.5944),
    "Mataram": (-8.5833, 116.1167),
}
MERCHANTS = {
    "UKM001": {"name": "Warung Makan Bu Sari", "type": "food", "city": "Surabaya"},
    "UKM002": {"name": "Toko Batik Pak Hendra", "type": "retail", "city": "Solo"},
    "UKM003": {"name": "Bengkel Motor Jaya", "type": "service", "city": "Bandung"},
    "UKM004": {"name": "Toko Sembako Maju", "type": "grocery", "city": "Medan"},
    "UKM005": {"name": "Salon Kecantikan Ayu", "type": "beauty", "city": "Jakarta"},
}
CITIES = list(CITY_COORDS.keys())
DAY_NAMES = {
    0: "Senin",
    1: "Selasa",
    2: "Rabu",
    3: "Kamis",
    4: "Jumat",
    5: "Sabtu",
    6: "Minggu",
}

# ── Session State ─────────────────────────────────────────────────────────────
for key, val in [
    ("orchestrator", None),
    ("cases", []),
    ("notifications", []),
    ("demo_running", False),
    ("batch_results", None),
]:
    if key not in st.session_state:
        st.session_state[key] = val

if st.session_state.orchestrator is None:
    st.session_state.orchestrator = AegisOrchestrator()


# ── Helpers ───────────────────────────────────────────────────────────────────
def make_gauge(score, title="Risk Score"):
    color = "#E74C3C" if score >= 0.65 else ("#E67E22" if score >= 0.40 else "#27AE60")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score * 100,
            title={"text": title, "font": {"size": 14, "color": "#aaa"}},
            number={"suffix": "%", "font": {"size": 22, "color": color}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#555"},
                "bar": {"color": color, "thickness": 0.3},
                "bgcolor": "#1a1d2e",
                "bordercolor": "#333",
                "steps": [
                    {"range": [0, 40], "color": "#0a1e0a"},
                    {"range": [40, 65], "color": "#1e1a0a"},
                    {"range": [65, 100], "color": "#1e0a0a"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 3},
                    "value": score * 100,
                },
            },
        )
    )
    fig.update_layout(height=180, margin=dict(l=20, r=20, t=40, b=10), **PLOTLY_DARK)
    return fig


def action_badge(action):
    if action == "BLOCK":
        return '<span class="badge-block">🚫 BLOCKED</span>'
    elif action == "STEP_UP_AUTH":
        return '<span class="badge-stepup">🔐 STEP-UP</span>'
    else:
        return '<span class="badge-approve">✅ APPROVED</span>'


def process_transaction(txn_dict):
    ml_result = predict(txn_dict)
    case = st.session_state.orchestrator.process(txn_dict, ml_result)
    st.session_state.cases.append(case)
    st.session_state.notifications.append(case["notification"])
    return case, ml_result


def normalize_city(loc):
    loc_str = str(loc).strip().title()
    if loc_str in CITY_COORDS:
        return loc_str
    for city in CITY_COORDS:
        if city.lower() in loc_str.lower():
            return city
    return "Jakarta"


def risk_color(level):
    return {
        "CRITICAL": "#E74C3C",
        "HIGH": "#E67E22",
        "MEDIUM": "#F39C12",
        "LOW": "#27AE60",
    }.get(level, "#aaa")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
    <div style='text-align:center; padding:20px 0 10px;'>
        <div style='font-size:36px'>⚔️</div>
        <div style='font-size:20px; font-weight:700; color:#F39C12;'>AEGIS SME</div>
        <div style='font-size:11px; color:#666; margin-top:4px;'>Autonomous Financial Guardian</div>
        <div style='font-size:10px; color:#444; margin-top:2px;'>Team Finvee · Varsity Hackathon 2026</div>
    </div>
    <hr style='border-color:#222; margin:10px 0;'>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("**System Status**")
    c1, c2 = st.columns(2)
    c1.metric("Cases", len(st.session_state.cases))
    c2.metric(
        "Alerts",
        len(
            [
                n
                for n in st.session_state.notifications
                if n.get("priority") in ["HIGH", "MEDIUM"]
            ]
        ),
    )

    if st.session_state.cases:
        actions = [c["final_action"] for c in st.session_state.cases]
        st.markdown(
            f"""
        <div style='background:#1a0a0a; border-radius:8px; padding:10px; margin-top:8px; font-size:12px;'>
            🚫 Blocked: <b style='color:#E74C3C'>{actions.count("BLOCK")}</b> &nbsp;|&nbsp;
            🔐 Step-Up: <b style='color:#E67E22'>{actions.count("STEP_UP_AUTH")}</b> &nbsp;|&nbsp;
            ✅ OK: <b style='color:#27AE60'>{actions.count("APPROVE")}</b>
        </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='border-color:#222;'>", unsafe_allow_html=True)
    st.markdown("**Quick Demo**")
    if st.button("▶ Run Demo Simulation (10 txn)", use_container_width=True):
        st.session_state.demo_running = True

    st.markdown("<hr style='border-color:#222;'>", unsafe_allow_html=True)
    st.markdown(
        """
    <div style='font-size:11px; color:#444; text-align:center;'>
        Case Study 2 · SDG 8<br>Fraud & Anomaly Detection<br>
        Agentic AI + LightGBM (AUC 0.9522)
    </div>""",
        unsafe_allow_html=True,
    )

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div style='padding:10px 0 20px;'>
    <h1 style='color:#F39C12; margin:0; font-size:28px;'>⚔️ AEGIS SME Dashboard</h1>
    <p style='color:#666; margin:4px 0 0; font-size:13px;'>
        Real-Time Autonomous Fraud Shield · Team Finvee · Varsity Hackathon 2026
    </p>
</div>""",
    unsafe_allow_html=True,
)

# ── Demo Simulation ───────────────────────────────────────────────────────────
if st.session_state.demo_running:
    st.session_state.demo_running = False
    with st.spinner("Running demo simulation..."):
        for i in range(10):
            is_fraud = random.random() < 0.3
            m_id = random.choice(list(MERCHANTS.keys()))
            m = MERCHANTS[m_id]
            if is_fraud:
                txn = {
                    "transaction_id": f"TXN_DEMO_{int(time.time() * 1000)}_{i}",
                    "merchant_id": m_id,
                    "merchant_type": m["type"],
                    "amount": round(random.uniform(3000000, 12000000), 2),
                    "hour": random.choice([1, 2, 3]),
                    "day_of_week": random.randint(0, 6),
                    "location": random.choice([c for c in CITIES if c != m["city"]]),
                    "device_id": f"DEV_{random.randint(500, 999):04d}",
                    "is_new_device": 1,
                    "transaction_count_1h": random.randint(8, 18),
                    "transaction_count_24h": random.randint(25, 50),
                    "amount_vs_avg_ratio": round(random.uniform(8, 18), 2),
                    "location_mismatch": 1,
                }
            else:
                txn = {
                    "transaction_id": f"TXN_DEMO_{int(time.time() * 1000)}_{i}",
                    "merchant_id": m_id,
                    "merchant_type": m["type"],
                    "amount": round(random.uniform(50000, 700000), 2),
                    "hour": random.randint(8, 20),
                    "day_of_week": random.randint(0, 6),
                    "location": m["city"],
                    "device_id": f"DEV_{random.randint(1, 10):04d}",
                    "is_new_device": 0,
                    "transaction_count_1h": random.randint(1, 5),
                    "transaction_count_24h": random.randint(5, 25),
                    "amount_vs_avg_ratio": round(random.uniform(0.5, 2.0), 2),
                    "location_mismatch": 0,
                }
            process_transaction(txn)
    st.success("Demo simulation complete! 10 transactions processed.")
    st.rerun()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "🔍 Analyze",
        "📂 Batch Upload",
        "🗺️ Fraud Map",
        "📈 Tracking",
        "📊 Live Dashboard",
        "🤖 Agent Workflow",
        "📲 Notifications",
    ]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Analyze Transaction
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Submit a Transaction for Analysis")
    col_form, col_result = st.columns([1, 1])

    with col_form:
        with st.form("analyze_form"):
            c1, c2 = st.columns(2)
            txn_id = c1.text_input("Transaction ID", value=f"TXN_{int(time.time())}")
            merchant_id = c2.selectbox(
                "Merchant",
                list(MERCHANTS.keys()),
                format_func=lambda k: f"{k} — {MERCHANTS[k]['name']}",
            )
            amount = c1.number_input(
                "Amount (Rp)", min_value=0.0, value=500000.0, step=10000.0
            )
            merchant_type = c2.selectbox(
                "Type", ["food", "retail", "service", "grocery", "beauty"]
            )
            hour = c1.slider("Hour", 0, 23, 14)
            location = c2.selectbox("Location", CITIES)

            c3, c4 = st.columns(2)
            is_new_device = c3.checkbox("New Device", value=False)
            location_mismatch = c4.checkbox("Location Mismatch", value=False)
            txn_1h = c3.number_input("Transactions (1h)", 0, 100, 2)
            txn_24h = c4.number_input("Transactions (24h)", 0, 500, 8)
            amt_ratio = st.slider("Amount vs Avg Ratio", 0.1, 30.0, 1.5, 0.1)
            submitted = st.form_submit_button(
                "⚡ Analyze Transaction", use_container_width=True
            )

    with col_result:
        if submitted:
            payload = {
                "transaction_id": txn_id,
                "merchant_id": merchant_id,
                "merchant_type": merchant_type,
                "amount": amount,
                "hour": hour,
                "day_of_week": 1,
                "location": location,
                "device_id": "DEV_NEW" if is_new_device else "DEV_0001",
                "is_new_device": int(is_new_device),
                "transaction_count_1h": txn_1h,
                "transaction_count_24h": txn_24h,
                "amount_vs_avg_ratio": amt_ratio,
                "location_mismatch": int(location_mismatch),
            }
            with st.spinner("🔄 Analyzing..."):
                case, ml = process_transaction(payload)

            score = case["risk_score"]
            level = case["risk_level"]
            action = case["final_action"]

            st.plotly_chart(make_gauge(score), use_container_width=True)
            st.markdown(action_badge(action), unsafe_allow_html=True)
            st.markdown(f"**Reasoning:** {case.get('reasoning', '')}")
            st.markdown("---")
            ca, cb, cc = st.columns(3)
            ca.metric("Processing Time", f"{case.get('processing_time_ms', 0):.0f}ms")
            cb.metric("LGB Score", f"{ml['lgb_score']:.4f}")
            cc.metric("Ensemble Score", f"{ml['ensemble_score']:.4f}")
            if case.get("notification", {}).get("message"):
                st.info(f"📱 **Notification:** {case['notification']['message']}")
        else:
            st.markdown(
                """
            <div style='text-align:center; padding:60px; color:#4a5568;'>
                <h1 style='font-size:4rem;'>⚔️</h1>
                <h3>AEGIS SME Guardian</h3>
                <p>Fill in transaction details and click Analyze.</p>
            </div>""",
                unsafe_allow_html=True,
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch Upload
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📂 Batch Transaction Upload")
    st.markdown(
        "Upload file **CSV** atau **Excel** berisi transaksi untuk dianalisis secara massal."
    )

    col_up, col_guide = st.columns([2, 1])

    with col_guide:
        st.markdown("#### 📋 Kolom yang Didukung")
        st.markdown("""
| Kolom | Wajib | Keterangan |
|---|---|---|
| `amount` | ✅ | Jumlah transaksi |
| `hour` | ✅ | Jam (0–23) |
| `transaction_id` | | ID unik |
| `merchant_id` | | ID merchant |
| `merchant_type` | | Jenis usaha |
| `location` | | Kota |
| `is_new_device` | | 0 / 1 |
| `location_mismatch` | | 0 / 1 |
| `amount_vs_avg_ratio` | | Rasio jumlah |
| `transaction_count_1h` | | Frekuensi 1 jam |
| `transaction_count_24h` | | Frekuensi 24 jam |
        """)
        with open(
            os.path.join(PROJECT_ROOT, "data/dummy/sample_transactions_template.csv"),
            "rb",
        ) as f:
            st.download_button(
                "⬇️ Download Template CSV",
                f.read(),
                "aegis_template.csv",
                "text/csv",
                use_container_width=True,
            )

    with col_up:
        uploaded_file = st.file_uploader(
            "Upload CSV atau Excel",
            type=["csv", "xlsx", "xls"],
            help="Maksimal 10.000 baris",
        )

        if uploaded_file is not None:
            st.success(
                f"✅ File diterima: **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)"
            )
            try:
                if uploaded_file.name.endswith(".csv"):
                    df_upload = pd.read_csv(uploaded_file)
                else:
                    df_upload = pd.read_excel(uploaded_file)

                st.markdown(f"**{len(df_upload)} transaksi** ditemukan. Preview:")
                st.dataframe(df_upload.head(5), use_container_width=True)

                if st.button(
                    "🚀 Analyze All Transactions",
                    use_container_width=True,
                    type="primary",
                ):
                    progress = st.progress(0, text="Memulai analisis...")
                    results_list = []
                    defaults = {
                        "transaction_id": None,
                        "merchant_id": "UKM001",
                        "merchant_type": "W",
                        "amount": 100.0,
                        "hour": 12,
                        "day_of_week": 1,
                        "location": "Jakarta",
                        "device_id": "DEV_0001",
                        "is_new_device": 0,
                        "transaction_count_1h": 2,
                        "transaction_count_24h": 8,
                        "amount_vs_avg_ratio": 1.0,
                        "location_mismatch": 0,
                    }
                    for i, (_, row) in enumerate(df_upload.iterrows()):
                        txn = {}
                        for col, default in defaults.items():
                            if col in row and pd.notna(row[col]):
                                txn[col] = row[col]
                            else:
                                txn[col] = (
                                    f"TXN_UPLOAD_{i:04d}"
                                    if col == "transaction_id"
                                    else default
                                )
                        txn["transaction_id"] = str(txn["transaction_id"])
                        try:
                            ml = predict(txn)
                            results_list.append(
                                {
                                    "transaction_id": txn["transaction_id"],
                                    "merchant_id": str(txn["merchant_id"]),
                                    "location": str(txn["location"]),
                                    "amount": float(txn["amount"]),
                                    "hour": int(txn["hour"]),
                                    "day_of_week": int(txn.get("day_of_week", 1)),
                                    "is_new_device": int(txn["is_new_device"]),
                                    "location_mismatch": int(txn["location_mismatch"]),
                                    "amount_vs_avg_ratio": float(
                                        txn["amount_vs_avg_ratio"]
                                    ),
                                    "transaction_count_1h": int(
                                        txn["transaction_count_1h"]
                                    ),
                                    "lgb_score": ml["lgb_score"],
                                    "ensemble_score": ml["ensemble_score"],
                                    "decision": ml["decision"],
                                    "risk_level": ml["risk_level"],
                                    "explanation": ml["explanation"],
                                }
                            )
                        except Exception as e:
                            results_list.append(
                                {
                                    "transaction_id": txn.get("transaction_id", str(i)),
                                    "merchant_id": str(txn.get("merchant_id", "")),
                                    "location": str(txn.get("location", "Jakarta")),
                                    "amount": float(txn.get("amount", 0)),
                                    "hour": int(txn.get("hour", 12)),
                                    "day_of_week": 1,
                                    "is_new_device": 0,
                                    "location_mismatch": 0,
                                    "amount_vs_avg_ratio": 1.0,
                                    "transaction_count_1h": 1,
                                    "lgb_score": 0.0,
                                    "ensemble_score": 0.0,
                                    "decision": "ERROR",
                                    "risk_level": "UNKNOWN",
                                    "explanation": str(e),
                                }
                            )
                        progress.progress(
                            (i + 1) / len(df_upload),
                            text=f"Analyzing {i + 1}/{len(df_upload)}...",
                        )

                    st.session_state.batch_results = pd.DataFrame(results_list)
                    progress.empty()
                    st.success(
                        f"✅ Analisis selesai! {len(results_list)} transaksi diproses."
                    )
                    st.rerun()
            except Exception as e:
                st.error(f"Gagal membaca file: {e}")

    # Show batch results
    if st.session_state.batch_results is not None:
        df_res = st.session_state.batch_results
        st.markdown("---")
        st.markdown("## 📊 Hasil Analisis Batch")

        total = len(df_res)
        blocked = (df_res["decision"] == "BLOCK").sum()
        step_up = (df_res["decision"] == "STEP_UP_AUTH").sum()
        approved = (df_res["decision"] == "APPROVE").sum()
        flagged_amt = df_res[df_res["decision"] != "APPROVE"]["amount"].sum()

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Transaksi", f"{total:,}")
        m2.metric(
            "🚫 Diblokir",
            f"{blocked}",
            f"{blocked / total * 100:.1f}%",
            delta_color="inverse",
        )
        m3.metric(
            "⚠️ Step-Up Auth",
            f"{step_up}",
            f"{step_up / total * 100:.1f}%",
            delta_color="off",
        )
        m4.metric("✅ Disetujui", f"{approved}", f"{approved / total * 100:.1f}%")
        m5.metric("💰 Jumlah Terblokir", f"Rp {flagged_amt:,.0f}")

        col_pie, col_tbl = st.columns([1, 2])
        with col_pie:
            fig_pie = px.pie(
                values=[blocked, step_up, approved],
                names=["Blocked", "Step-Up Auth", "Approved"],
                color_discrete_sequence=["#E74C3C", "#E67E22", "#27AE60"],
                title="Distribusi Keputusan",
            )
            fig_pie.update_layout(height=300, **PLOTLY_DARK)
            st.plotly_chart(fig_pie, use_container_width=True)
        with col_tbl:
            st.markdown("#### Detail Hasil")
            disp = df_res[
                [
                    "transaction_id",
                    "merchant_id",
                    "location",
                    "amount",
                    "ensemble_score",
                    "decision",
                    "risk_level",
                ]
            ].copy()
            disp["amount"] = disp["amount"].apply(lambda x: f"Rp {x:,.0f}")
            disp["ensemble_score"] = disp["ensemble_score"].apply(lambda x: f"{x:.4f}")
            st.dataframe(disp, use_container_width=True, height=280)

        fig_hist = px.histogram(
            df_res,
            x="ensemble_score",
            nbins=30,
            color="decision",
            color_discrete_map={
                "BLOCK": "#E74C3C",
                "STEP_UP_AUTH": "#E67E22",
                "APPROVE": "#27AE60",
                "ERROR": "#888",
            },
            title="Distribusi Risk Score",
            labels={"ensemble_score": "Ensemble Risk Score"},
        )
        fig_hist.update_layout(height=280, **PLOTLY_DARK)
        st.plotly_chart(fig_hist, use_container_width=True)

        csv_export = df_res.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Export Hasil ke CSV",
            csv_export,
            f"aegis_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            use_container_width=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Fraud Map
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🗺️ Fraud Geographic Map")
    st.markdown("Visualisasi distribusi dan konsentrasi fraud per kota di Indonesia.")

    map_source = st.radio(
        "Sumber Data",
        ["Generate Sample Data", "Batch Upload Results", "Demo Simulation"],
        horizontal=True,
    )

    map_df = None
    if (
        map_source == "Batch Upload Results"
        and st.session_state.batch_results is not None
    ):
        map_df = st.session_state.batch_results.copy()
    elif map_source == "Demo Simulation" and st.session_state.cases:
        rows = []
        for c in st.session_state.cases:
            rows.append(
                {
                    "transaction_id": c.get("case_id", ""),
                    "merchant_id": c.get("merchant_id", "UKM001"),
                    "location": random.choice(CITIES),
                    "amount": c.get("amount", random.uniform(50000, 5000000)),
                    "hour": random.randint(0, 23),
                    "day_of_week": random.randint(0, 6),
                    "is_new_device": 1 if c.get("final_action") == "BLOCK" else 0,
                    "location_mismatch": 1 if c.get("final_action") == "BLOCK" else 0,
                    "amount_vs_avg_ratio": 15.0
                    if c.get("final_action") == "BLOCK"
                    else 1.5,
                    "transaction_count_1h": 15
                    if c.get("final_action") == "BLOCK"
                    else 2,
                    "lgb_score": c.get("risk_score", 0.1),
                    "ensemble_score": c.get("risk_score", 0.1),
                    "decision": c.get("final_action", "APPROVE"),
                    "risk_level": c.get("risk_level", "LOW"),
                    "explanation": "",
                }
            )
        map_df = pd.DataFrame(rows)
    else:
        # Generate sample
        np.random.seed(42)
        sample_rows = []
        for i in range(300):
            city = random.choice(CITIES)
            is_fraud = random.random() < 0.25
            sample_rows.append(
                {
                    "transaction_id": f"SAMPLE_{i:04d}",
                    "merchant_id": f"UKM{random.randint(1, 5):03d}",
                    "location": city,
                    "amount": random.uniform(5000, 15000)
                    if is_fraud
                    else random.uniform(50, 2000),
                    "hour": random.choice([1, 2, 3, 4])
                    if is_fraud
                    else random.randint(8, 20),
                    "day_of_week": random.randint(0, 6),
                    "is_new_device": 1 if is_fraud else 0,
                    "location_mismatch": 1 if is_fraud else 0,
                    "amount_vs_avg_ratio": random.uniform(10, 25)
                    if is_fraud
                    else random.uniform(0.5, 2.5),
                    "transaction_count_1h": random.randint(10, 20)
                    if is_fraud
                    else random.randint(1, 5),
                    "lgb_score": random.uniform(0.3, 0.9)
                    if is_fraud
                    else random.uniform(0.01, 0.15),
                    "ensemble_score": random.uniform(0.4, 0.95)
                    if is_fraud
                    else random.uniform(0.02, 0.2),
                    "decision": "BLOCK"
                    if is_fraud and random.random() > 0.3
                    else ("STEP_UP_AUTH" if is_fraud else "APPROVE"),
                    "risk_level": "CRITICAL" if is_fraud else "LOW",
                    "explanation": "",
                }
            )
        map_df = pd.DataFrame(sample_rows)

    if map_df is not None and not map_df.empty:
        map_df["city"] = map_df["location"].apply(normalize_city)
        city_sum = (
            map_df.groupby("city")
            .agg(
                total=("transaction_id", "count"),
                blocked=("decision", lambda x: (x == "BLOCK").sum()),
                step_up=("decision", lambda x: (x == "STEP_UP_AUTH").sum()),
                approved=("decision", lambda x: (x == "APPROVE").sum()),
                avg_score=("ensemble_score", "mean"),
                total_amount=("amount", "sum"),
            )
            .reset_index()
        )
        city_sum["fraud_rate"] = (city_sum["blocked"] / city_sum["total"] * 100).round(
            1
        )
        city_sum["suspicious_rate"] = (
            (city_sum["blocked"] + city_sum["step_up"]) / city_sum["total"] * 100
        ).round(1)
        city_sum["lat"] = city_sum["city"].map(
            lambda c: CITY_COORDS.get(c, (-6.2, 106.8))[0]
        )
        city_sum["lon"] = city_sum["city"].map(
            lambda c: CITY_COORDS.get(c, (-6.2, 106.8))[1]
        )

        col_map, col_rank = st.columns([3, 1])
        with col_map:
            fig_map = px.scatter_mapbox(
                city_sum,
                lat="lat",
                lon="lon",
                size="total",
                color="fraud_rate",
                color_continuous_scale=["#27AE60", "#F39C12", "#E67E22", "#E74C3C"],
                range_color=[0, 100],
                hover_name="city",
                hover_data={
                    "total": True,
                    "blocked": True,
                    "fraud_rate": True,
                    "avg_score": ":.4f",
                    "lat": False,
                    "lon": False,
                },
                size_max=50,
                zoom=4,
                center={"lat": -2.5, "lon": 118.0},
                mapbox_style="carto-darkmatter",
                title="🗺️ Fraud Distribution Map — Indonesia",
                labels={
                    "fraud_rate": "Fraud Rate (%)",
                    "total": "Total Txn",
                    "blocked": "Blocked",
                },
            )
            fig_map.update_layout(
                height=480,
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ccc"),
                coloraxis_colorbar=dict(
                    title="Fraud Rate %", tickfont=dict(color="#ccc")
                ),
                margin=dict(t=50, b=0, l=0, r=0),
            )
            st.plotly_chart(fig_map, use_container_width=True)

        with col_rank:
            st.markdown("#### 🏙️ Top Cities by Fraud")
            top_cities = city_sum.nlargest(8, "blocked")[
                ["city", "total", "blocked", "fraud_rate"]
            ]
            for _, row in top_cities.iterrows():
                c = (
                    "#E74C3C"
                    if row["fraud_rate"] > 50
                    else ("#E67E22" if row["fraud_rate"] > 25 else "#F39C12")
                )
                st.markdown(
                    f"""
                <div style='background:#1a1d2e; border-left:3px solid {c}; padding:8px; margin:4px 0; border-radius:4px;'>
                    <b style='color:#ddd'>{row["city"]}</b><br>
                    <span style='color:{c}'>🚫 {int(row["blocked"])} blocked ({row["fraud_rate"]:.0f}%)</span><br>
                    <span style='color:#666; font-size:0.8rem'>{int(row["total"])} total txn</span>
                </div>""",
                    unsafe_allow_html=True,
                )

        col_bar, col_stack = st.columns(2)
        with col_bar:
            fig_bar = px.bar(
                city_sum.sort_values("fraud_rate", ascending=True).tail(12),
                x="fraud_rate",
                y="city",
                orientation="h",
                color="fraud_rate",
                color_continuous_scale=["#27AE60", "#F39C12", "#E67E22", "#E74C3C"],
                title="Fraud Rate per Kota (%)",
                labels={"fraud_rate": "Fraud Rate (%)", "city": "Kota"},
            )
            fig_bar.update_layout(height=350, **PLOTLY_DARK, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_stack:
            pivot = (
                map_df.groupby(["city", "decision"])
                .size()
                .unstack(fill_value=0)
                .reset_index()
            )
            for col in ["APPROVE", "STEP_UP_AUTH", "BLOCK"]:
                if col not in pivot.columns:
                    pivot[col] = 0
            pivot_melted = pivot.melt(
                id_vars="city", value_vars=["APPROVE", "STEP_UP_AUTH", "BLOCK"]
            )
            pivot_melted = pivot_melted.rename(columns={"variable": "decision"})
            fig_stack = px.bar(
                pivot_melted,
                x="city",
                y="value",
                color="decision",
                color_discrete_map={
                    "BLOCK": "#E74C3C",
                    "STEP_UP_AUTH": "#E67E22",
                    "APPROVE": "#27AE60",
                },
                title="Distribusi Keputusan per Kota",
                labels={"value": "Jumlah", "city": "Kota", "decision": "Keputusan"},
                barmode="stack",
            )
            fig_stack.update_layout(height=350, **PLOTLY_DARK)
            st.plotly_chart(fig_stack, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Temporal Tracking
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 📈 Temporal Fraud Tracking")
    st.markdown(
        "Analisis tren fraud berdasarkan waktu — kapan, seberapa sering, dan seberapa besar."
    )

    track_source = st.radio(
        "Sumber Data",
        ["Generate Sample Data", "Batch Upload Results"],
        horizontal=True,
        key="track_src",
    )

    track_df = None
    if (
        track_source == "Batch Upload Results"
        and st.session_state.batch_results is not None
    ):
        track_df = st.session_state.batch_results.copy()
    else:
        np.random.seed(123)
        rows = []
        for i in range(500):
            is_fraud = random.random() < 0.22
            hour = (
                random.choice([1, 2, 3, 23])
                if is_fraud
                else random.choices(
                    range(24),
                    weights=[
                        1,
                        1,
                        1,
                        1,
                        1,
                        2,
                        3,
                        5,
                        8,
                        10,
                        10,
                        9,
                        8,
                        8,
                        9,
                        10,
                        10,
                        9,
                        8,
                        6,
                        5,
                        4,
                        3,
                        2,
                    ],
                )[0]
            )
            rows.append(
                {
                    "transaction_id": f"T{i:04d}",
                    "merchant_id": f"UKM{random.randint(1, 5):03d}",
                    "location": random.choice(CITIES),
                    "amount": random.uniform(3000, 15000)
                    if is_fraud
                    else random.uniform(50, 2000),
                    "hour": hour,
                    "day_of_week": random.randint(0, 6),
                    "is_new_device": 1 if is_fraud else 0,
                    "location_mismatch": 1 if is_fraud else 0,
                    "amount_vs_avg_ratio": random.uniform(10, 25)
                    if is_fraud
                    else random.uniform(0.5, 3),
                    "transaction_count_1h": random.randint(10, 20)
                    if is_fraud
                    else random.randint(1, 5),
                    "lgb_score": random.uniform(0.3, 0.9)
                    if is_fraud
                    else random.uniform(0.01, 0.15),
                    "ensemble_score": random.uniform(0.4, 0.95)
                    if is_fraud
                    else random.uniform(0.02, 0.2),
                    "decision": "BLOCK"
                    if is_fraud and random.random() > 0.2
                    else ("STEP_UP_AUTH" if is_fraud else "APPROVE"),
                    "risk_level": "CRITICAL" if is_fraud else "LOW",
                    "explanation": "",
                }
            )
        track_df = pd.DataFrame(rows)

    if track_df is not None and not track_df.empty:
        total = len(track_df)
        blocked = (track_df["decision"] == "BLOCK").sum()
        step_up = (track_df["decision"] == "STEP_UP_AUTH").sum()
        approved = (track_df["decision"] == "APPROVE").sum()
        avg_score = track_df["ensemble_score"].mean()
        flagged_amt = track_df[track_df["decision"] != "APPROVE"]["amount"].sum()

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Total Transaksi", f"{total:,}")
        k2.metric("🚫 Blocked", f"{blocked}", f"{blocked / total * 100:.1f}%")
        k3.metric("⚠️ Step-Up", f"{step_up}", f"{step_up / total * 100:.1f}%")
        k4.metric("✅ Approved", f"{approved}", f"{approved / total * 100:.1f}%")
        k5.metric("Avg Risk Score", f"{avg_score:.4f}")
        k6.metric("💰 Flagged Amt", f"Rp {flagged_amt:,.0f}")

        st.markdown("---")

        # Hourly + Daily
        col_h, col_d = st.columns(2)
        with col_h:
            hourly = (
                track_df.groupby("hour")
                .agg(
                    total=("transaction_id", "count"),
                    blocked=("decision", lambda x: (x == "BLOCK").sum()),
                    avg_score=("ensemble_score", "mean"),
                )
                .reset_index()
            )
            hourly["fraud_rate"] = (hourly["blocked"] / hourly["total"] * 100).round(1)

            fig_h = make_subplots(specs=[[{"secondary_y": True}]])
            fig_h.add_trace(
                go.Bar(
                    x=hourly["hour"],
                    y=hourly["total"],
                    name="Total",
                    marker_color="#1e3a5f",
                    opacity=0.8,
                ),
                secondary_y=False,
            )
            fig_h.add_trace(
                go.Scatter(
                    x=hourly["hour"],
                    y=hourly["fraud_rate"],
                    name="Fraud Rate %",
                    line=dict(color="#E74C3C", width=2.5),
                    mode="lines+markers",
                    marker=dict(size=6),
                ),
                secondary_y=True,
            )
            fig_h.update_layout(
                title="Transaksi & Fraud Rate per Jam",
                height=300,
                **PLOTLY_DARK,
                legend=dict(orientation="h", y=-0.25),
            )
            fig_h.update_xaxes(title_text="Jam", gridcolor="#2a2d3e")
            fig_h.update_yaxes(
                title_text="Jumlah", secondary_y=False, gridcolor="#2a2d3e"
            )
            fig_h.update_yaxes(
                title_text="Fraud Rate %", secondary_y=True, gridcolor="#2a2d3e"
            )
            st.plotly_chart(fig_h, use_container_width=True)

        with col_d:
            daily = (
                track_df.groupby("day_of_week")
                .agg(
                    total=("transaction_id", "count"),
                    blocked=("decision", lambda x: (x == "BLOCK").sum()),
                )
                .reset_index()
            )
            daily["day_name"] = daily["day_of_week"].map(DAY_NAMES)
            daily["fraud_rate"] = (daily["blocked"] / daily["total"] * 100).round(1)

            fig_d = make_subplots(specs=[[{"secondary_y": True}]])
            fig_d.add_trace(
                go.Bar(
                    x=daily["day_name"],
                    y=daily["total"],
                    name="Total",
                    marker_color="#1e3a5f",
                    opacity=0.8,
                ),
                secondary_y=False,
            )
            fig_d.add_trace(
                go.Scatter(
                    x=daily["day_name"],
                    y=daily["fraud_rate"],
                    name="Fraud Rate %",
                    line=dict(color="#E67E22", width=2.5),
                    mode="lines+markers",
                    marker=dict(size=6),
                ),
                secondary_y=True,
            )
            fig_d.update_layout(
                title="Transaksi & Fraud Rate per Hari",
                height=300,
                **PLOTLY_DARK,
                legend=dict(orientation="h", y=-0.25),
            )
            fig_d.update_xaxes(title_text="Hari", gridcolor="#2a2d3e")
            fig_d.update_yaxes(
                title_text="Jumlah", secondary_y=False, gridcolor="#2a2d3e"
            )
            fig_d.update_yaxes(
                title_text="Fraud Rate %", secondary_y=True, gridcolor="#2a2d3e"
            )
            st.plotly_chart(fig_d, use_container_width=True)

        # Heatmap Hour × Day
        st.markdown("#### 🔥 Fraud Heatmap — Jam × Hari")
        hm = (
            track_df[track_df["decision"] == "BLOCK"]
            .groupby(["day_of_week", "hour"])
            .size()
            .reset_index(name="count")
        )
        hm_pivot = hm.pivot(index="day_of_week", columns="hour", values="count").fillna(
            0
        )
        hm_pivot.index = [DAY_NAMES.get(i, str(i)) for i in hm_pivot.index]
        for h in range(24):
            if h not in hm_pivot.columns:
                hm_pivot[h] = 0
        hm_pivot = hm_pivot[sorted(hm_pivot.columns)]

        fig_hm = px.imshow(
            hm_pivot,
            color_continuous_scale=["#0f1117", "#1e3a5f", "#E67E22", "#E74C3C"],
            title="Konsentrasi Fraud per Jam dan Hari",
            labels={"x": "Jam", "y": "Hari", "color": "Jumlah Fraud"},
            aspect="auto",
        )
        fig_hm.update_layout(height=280, **PLOTLY_DARK)
        st.plotly_chart(fig_hm, use_container_width=True)

        # Violin + Scatter
        col_v, col_s = st.columns(2)
        with col_v:
            fig_vio = px.violin(
                track_df,
                y="ensemble_score",
                x="decision",
                color="decision",
                color_discrete_map={
                    "BLOCK": "#E74C3C",
                    "STEP_UP_AUTH": "#E67E22",
                    "APPROVE": "#27AE60",
                },
                title="Distribusi Risk Score per Keputusan",
                box=True,
                points="outliers",
            )
            fig_vio.update_layout(height=320, **PLOTLY_DARK, showlegend=False)
            st.plotly_chart(fig_vio, use_container_width=True)

        with col_s:
            fig_sc = px.scatter(
                track_df.sample(min(300, len(track_df))),
                x="amount",
                y="ensemble_score",
                color="decision",
                color_discrete_map={
                    "BLOCK": "#E74C3C",
                    "STEP_UP_AUTH": "#E67E22",
                    "APPROVE": "#27AE60",
                },
                title="Amount vs Risk Score",
                labels={"amount": "Jumlah (Rp)", "ensemble_score": "Risk Score"},
                opacity=0.7,
            )
            fig_sc.update_layout(height=320, **PLOTLY_DARK)
            st.plotly_chart(fig_sc, use_container_width=True)

        # Top risky transactions
        st.markdown("#### 🚨 Top 10 Transaksi Paling Berisiko")
        top10 = track_df.nlargest(10, "ensemble_score")[
            [
                "transaction_id",
                "merchant_id",
                "location",
                "amount",
                "hour",
                "ensemble_score",
                "decision",
                "risk_level",
            ]
        ].copy()
        top10["amount"] = top10["amount"].apply(lambda x: f"Rp {x:,.0f}")
        top10["ensemble_score"] = top10["ensemble_score"].apply(lambda x: f"{x:.4f}")
        top10["hour"] = top10["hour"].apply(lambda x: f"{x:02d}:00")
        st.dataframe(top10, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Live Dashboard
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### 📊 Live Dashboard")

    if not st.session_state.cases:
        st.info("ℹ️ Belum ada transaksi diproses. Jalankan Demo Simulation di sidebar.")
    else:
        cases = st.session_state.cases
        actions = [c.get("final_action", "APPROVE") for c in cases]
        scores = [c.get("ml_result", {}).get("ensemble_score", 0.0) for c in cases]
        blocked = actions.count("BLOCK")
        step_up = actions.count("STEP_UP_AUTH")
        approved = actions.count("APPROVE")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Transactions", len(cases))
        m2.metric(
            "🚫 Blocked",
            blocked,
            f"{blocked / len(cases) * 100:.1f}%",
            delta_color="inverse",
        )
        m3.metric(
            "⚠️ Step-Up",
            step_up,
            f"{step_up / len(cases) * 100:.1f}%",
            delta_color="off",
        )
        m4.metric("✅ Approved", approved, f"{approved / len(cases) * 100:.1f}%")
        m5.metric("Avg Risk Score", f"{np.mean(scores):.4f}")

        col_pie2, col_bar2 = st.columns(2)
        with col_pie2:
            fig2 = px.pie(
                values=[blocked, step_up, approved],
                names=["Blocked", "Step-Up Auth", "Approved"],
                color_discrete_sequence=["#E74C3C", "#E67E22", "#27AE60"],
                title="Decision Distribution",
            )
            fig2.update_layout(height=300, **PLOTLY_DARK)
            st.plotly_chart(fig2, use_container_width=True)

        with col_bar2:
            df_cases = pd.DataFrame(
                [
                    {
                        "id": c["case_id"],
                        "score": c.get("ml_result", {}).get("ensemble_score", 0.0),
                        "action": c.get("final_action", "APPROVE"),
                    }
                    for c in cases[-15:]
                ]
            )
            fig_scores = px.bar(
                df_cases,
                x="id",
                y="score",
                color="action",
                color_discrete_map={
                    "BLOCK": "#E74C3C",
                    "STEP_UP_AUTH": "#E67E22",
                    "APPROVE": "#27AE60",
                },
                title="Recent Transaction Risk Scores",
                labels={"id": "Transaction", "score": "Risk Score"},
            )
            fig_scores.update_layout(height=300, **PLOTLY_DARK, xaxis_tickangle=45)
            st.plotly_chart(fig_scores, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Agent Workflow
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("### 🤖 Agent Workflow Visualization")

    if not st.session_state.cases:
        st.info("ℹ️ Belum ada case. Jalankan Demo Simulation atau Analyze Transaction.")
    else:
        case_ids = [c["case_id"] for c in st.session_state.cases]
        sel_id = st.selectbox("Pilih Case", case_ids[::-1])
        sel_case = next(
            (c for c in st.session_state.cases if c["case_id"] == sel_id), None
        )

        if sel_case:
            action = sel_case.get("final_action", "APPROVE")
            color = {
                "BLOCK": "#E74C3C",
                "STEP_UP_AUTH": "#E67E22",
                "APPROVE": "#27AE60",
            }.get(action, "#aaa")
            st.markdown(
                f"""
            <div style='background:#1a1d2e; border:1px solid {color}; border-radius:12px; padding:20px; margin-bottom:20px;'>
                <h3 style='color:{color}; margin:0'>{action_badge(action)} {action}</h3>
                <p style='color:#666; margin:4px 0'>Case: {sel_case["case_id"]} | Score: {sel_case.get("ml_result", {}).get("ensemble_score", 0.0):.4f}</p>
            </div>""",
                unsafe_allow_html=True,
            )

            agents_info = [
                (
                    "🔔 Monitor Agent",
                    "Mendeteksi anomali & trigger investigasi",
                    "#4A90D9",
                ),
                (
                    "🕵️ Investigator Agent",
                    "5 tool calls: device, lokasi, pola, riwayat",
                    "#8E44AD",
                ),
                ("⚖️ Resolution Agent", "LLM reasoning → keputusan final", color),
                (
                    "📱 Communicator Agent",
                    "Notifikasi personal ke pemilik UKM",
                    "#27AE60",
                ),
            ]
            cols = st.columns(4)
            for col, (name, desc, bg) in zip(cols, agents_info):
                col.markdown(
                    f"""
                <div style='background:#1a1d2e; border-top:3px solid {bg}; border-radius:10px; padding:15px; text-align:center; min-height:90px;'>
                    <div style='font-size:1.4rem'>{name.split()[0]}</div>
                    <div style='font-size:0.85rem; color:#ddd; font-weight:600'>{" ".join(name.split()[1:])}</div>
                    <div style='font-size:0.75rem; color:#666; margin-top:4px'>{desc}</div>
                </div>""",
                    unsafe_allow_html=True,
                )

            st.markdown("---")
            mon = sel_case.get("monitor_result", {})
            inv = sel_case.get("investigator_result", {})
            res = sel_case.get("resolution_result", {})
            notif = sel_case.get("notification", {})

            steps = [
                (
                    "🔔 1. Monitor Agent",
                    "#4A90D9",
                    f"**Action:** `{mon.get('action', 'N/A')}`\n\n**Reason:** {mon.get('trigger_reason', 'N/A')}",
                ),
                (
                    "🕵️ 2. Investigator Agent",
                    "#8E44AD",
                    f"**Tools:** {', '.join([f'`{t}`' for t in inv.get('tools_called', [])])}\n\n**Device:** {'⚠️ Unknown' if not inv.get('device_check', {}).get('is_known_device') else '✅ Known'} | **Location:** {'⚠️ Mismatch' if inv.get('location_check', {}).get('mismatch') else '✅ OK'}\n\n**Fraud Confidence:** {inv.get('fraud_confidence', 0):.0%}",
                ),
                (
                    "⚖️ 3. Resolution Agent",
                    color,
                    f"**Decision:** `{res.get('final_action', 'N/A')}`\n\n> {res.get('reasoning', 'N/A')}",
                ),
                (
                    "📱 4. Communicator Agent",
                    "#27AE60",
                    f"**Recipient:** {notif.get('recipient', 'N/A')} | **Priority:** `{notif.get('priority', 'N/A')}`\n\n> {notif.get('message', 'N/A')}",
                ),
            ]
            for name, c, content in steps:
                with st.expander(name, expanded=True):
                    st.markdown(content)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — Notifications
# ══════════════════════════════════════════════════════════════════════════════
with tab7:
    st.markdown("### 📲 Notification Center")
    st.markdown("Semua notifikasi yang dikirim ke pemilik UKM oleh Communicator Agent.")

    if not st.session_state.notifications:
        st.info(
            "ℹ️ Belum ada notifikasi. Jalankan Demo Simulation atau Analyze Transaction."
        )
    else:
        filter_p = st.multiselect(
            "Filter Prioritas",
            ["HIGH", "MEDIUM", "LOW"],
            default=["HIGH", "MEDIUM", "LOW"],
        )
        filtered = [
            n
            for n in reversed(st.session_state.notifications)
            if n.get("priority") in filter_p
        ]
        st.markdown(f"Menampilkan **{len(filtered)}** notifikasi")

        for notif in filtered:
            priority = notif.get("priority", "LOW")
            icon = (
                "🚨" if priority == "HIGH" else ("⚠️" if priority == "MEDIUM" else "✅")
            )
            st.markdown(
                f"""
            <div class='notification-card {priority}'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <span style='font-weight:600; color:#ddd;'>{icon} {notif.get("recipient", "Owner")}</span>
                    <span style='font-size:11px; color:#555;'>{notif.get("timestamp", "")[:19]}</span>
                </div>
                <div style='font-size:11px; color:#666; margin:4px 0;'>
                    {notif.get("merchant_id", "")} · {notif.get("notification_type", "")} · {notif.get("channel", "")}
                </div>
                <div style='color:#ccc; font-size:13px; margin-top:8px; line-height:1.5;'>
                    {notif.get("message", "")}
                </div>
                <div style='margin-top:8px;'>
                    {"".join([f"<span style='background:#1e2130; border:1px solid #333; border-radius:4px; padding:3px 8px; font-size:11px; margin-right:6px; color:#aaa;'>{b}</span>" for b in notif.get("action_buttons", [])])}
                </div>
            </div>""",
                unsafe_allow_html=True,
            )
