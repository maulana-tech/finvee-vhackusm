"""
AEGIS SME — Streamlit Dashboard
Team Finvee | Varsity Hackathon 2026

Real-time fraud monitoring dashboard for SME owners and analysts.
Features:
  - Live transaction analysis
  - Agent workflow visualization
  - Risk score gauges
  - Notification center
  - Demo simulation mode
"""

import sys
import os
sys.path.insert(0, "/home/ubuntu/aegis-sme")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
import random
import json
from datetime import datetime

from ml.predictor import predict
from agents.aegis_agents import (
    AegisOrchestrator, CASE_LOG, NOTIFICATION_LOG, TRANSACTION_HISTORY
)

# ─── Page Config ───
st.set_page_config(
    page_title="AEGIS SME | Finvee",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background-color: #0f1117; }

    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252840);
        border-radius: 12px;
        padding: 20px;
        border-left: 4px solid #4A90D9;
        margin-bottom: 10px;
    }
    .metric-card.red   { border-left-color: #E74C3C; }
    .metric-card.green { border-left-color: #27AE60; }
    .metric-card.orange{ border-left-color: #E67E22; }
    .metric-card.purple{ border-left-color: #8E44AD; }

    .agent-step {
        background: #1a1d2e;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 6px 0;
        border-left: 3px solid #4A90D9;
        font-size: 13px;
    }
    .agent-step.active { border-left-color: #F39C12; background: #1e1a0a; }
    .agent-step.done   { border-left-color: #27AE60; background: #0a1e0a; }

    .notification-card {
        background: #1a1d2e;
        border-radius: 10px;
        padding: 14px;
        margin: 8px 0;
        border: 1px solid #2a2d3e;
    }
    .notification-card.HIGH   { border-color: #E74C3C; }
    .notification-card.MEDIUM { border-color: #E67E22; }
    .notification-card.LOW    { border-color: #27AE60; }

    .badge-block    { background:#E74C3C; color:white; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:600; }
    .badge-stepup   { background:#E67E22; color:white; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:600; }
    .badge-approve  { background:#27AE60; color:white; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:600; }

    .stButton > button {
        background: linear-gradient(135deg, #4A90D9, #8E44AD);
        color: white; border: none; border-radius: 8px;
        padding: 10px 24px; font-weight: 600;
        transition: all 0.3s;
    }
    .stButton > button:hover { opacity: 0.9; transform: translateY(-1px); }

    div[data-testid="stSidebarContent"] { background: #0d1117; }
    .stTabs [data-baseweb="tab"] { color: #aaa; }
    .stTabs [aria-selected="true"] { color: #4A90D9 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Initialize Session State ───
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = AegisOrchestrator()
if "cases" not in st.session_state:
    st.session_state.cases = []
if "notifications" not in st.session_state:
    st.session_state.notifications = []
if "demo_running" not in st.session_state:
    st.session_state.demo_running = False

MERCHANTS = {
    "UKM001": {"name": "Warung Makan Bu Sari", "type": "food", "city": "Surabaya"},
    "UKM002": {"name": "Toko Batik Pak Hendra", "type": "retail", "city": "Solo"},
    "UKM003": {"name": "Bengkel Motor Jaya", "type": "service", "city": "Bandung"},
    "UKM004": {"name": "Toko Sembako Maju", "type": "grocery", "city": "Medan"},
    "UKM005": {"name": "Salon Kecantikan Ayu", "type": "beauty", "city": "Jakarta"},
}
CITIES = ["Jakarta", "Surabaya", "Bandung", "Medan", "Solo", "Makassar", "Palembang"]

# ─── Helper Functions ───
def make_gauge(score, title="Risk Score"):
    color = "#E74C3C" if score >= 0.75 else ("#E67E22" if score >= 0.45 else "#27AE60")
    fig = go.Figure(go.Indicator(
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
                {"range": [0, 45], "color": "#0a1e0a"},
                {"range": [45, 75], "color": "#1e1a0a"},
                {"range": [75, 100], "color": "#1e0a0a"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "value": score * 100}
        }
    ))
    fig.update_layout(
        height=180, margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#ccc"}
    )
    return fig

def action_badge(action):
    if action == "BLOCK":
        return '<span class="badge-block">🚫 BLOCKED</span>'
    elif action == "STEP_UP_AUTH":
        return '<span class="badge-stepup">🔐 STEP-UP</span>'
    else:
        return '<span class="badge-approve">✅ APPROVED</span>'

def process_transaction(txn_dict):
    """Run full pipeline and store results."""
    ml_result = predict(txn_dict)
    case = st.session_state.orchestrator.process(txn_dict, ml_result)
    st.session_state.cases.append(case)
    st.session_state.notifications.append(case["notification"])
    return case, ml_result

# ─── SIDEBAR ───
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px;'>
        <div style='font-size:36px'>⚔️</div>
        <div style='font-size:20px; font-weight:700; color:#F39C12;'>AEGIS SME</div>
        <div style='font-size:11px; color:#666; margin-top:4px;'>Autonomous Financial Guardian</div>
        <div style='font-size:10px; color:#444; margin-top:2px;'>Team Finvee · Varsity Hackathon 2026</div>
    </div>
    <hr style='border-color:#222; margin:10px 0;'>
    """, unsafe_allow_html=True)

    st.markdown("**System Status**")
    col1, col2 = st.columns(2)
    col1.metric("Cases", len(st.session_state.cases))
    col2.metric("Alerts", len([n for n in st.session_state.notifications if n.get("priority") in ["HIGH","MEDIUM"]]))

    if st.session_state.cases:
        actions = [c["final_action"] for c in st.session_state.cases]
        blocked = actions.count("BLOCK")
        st.markdown(f"""
        <div style='background:#1a0a0a; border-radius:8px; padding:10px; margin-top:8px; font-size:12px;'>
            🚫 Blocked: <b style='color:#E74C3C'>{blocked}</b> &nbsp;|&nbsp;
            🔐 Step-Up: <b style='color:#E67E22'>{actions.count("STEP_UP_AUTH")}</b> &nbsp;|&nbsp;
            ✅ OK: <b style='color:#27AE60'>{actions.count("APPROVE")}</b>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#222;'>", unsafe_allow_html=True)
    st.markdown("**Quick Demo**")
    if st.button("▶ Run Demo Simulation (10 txn)", use_container_width=True):
        st.session_state.demo_running = True

    st.markdown("<hr style='border-color:#222;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:11px; color:#444; text-align:center;'>
        Case Study 2 · SDG 8<br>
        Fraud & Anomaly Detection<br>
        Agentic AI + LightGBM + Autoencoder
    </div>
    """, unsafe_allow_html=True)

# ─── MAIN HEADER ───
st.markdown("""
<div style='padding: 10px 0 20px;'>
    <h1 style='color:#F39C12; margin:0; font-size:28px;'>⚔️ AEGIS SME Dashboard</h1>
    <p style='color:#666; margin:4px 0 0; font-size:13px;'>
        Real-Time Autonomous Fraud Shield · Team Finvee · Varsity Hackathon 2026
    </p>
</div>
""", unsafe_allow_html=True)

# ─── DEMO SIMULATION ───
if st.session_state.demo_running:
    st.session_state.demo_running = False
    with st.spinner("Running demo simulation..."):
        for i in range(10):
            is_fraud = random.random() < 0.3
            m_id = random.choice(list(MERCHANTS.keys()))
            m = MERCHANTS[m_id]
            if is_fraud:
                txn = {
                    "transaction_id": f"TXN_DEMO_{int(time.time()*1000)}_{i}",
                    "merchant_id": m_id, "merchant_type": m["type"],
                    "amount": round(random.uniform(3000000, 12000000), 2),
                    "hour": random.choice([1, 2, 3]),
                    "day_of_week": random.randint(0, 6),
                    "location": random.choice([c for c in CITIES if c != m["city"]]),
                    "device_id": f"DEV_{random.randint(500,999):04d}",
                    "is_new_device": 1, "transaction_count_1h": 1,
                    "transaction_count_24h": 4,
                    "amount_vs_avg_ratio": round(random.uniform(8, 18), 2),
                    "location_mismatch": 1
                }
            else:
                txn = {
                    "transaction_id": f"TXN_DEMO_{int(time.time()*1000)}_{i}",
                    "merchant_id": m_id, "merchant_type": m["type"],
                    "amount": round(random.uniform(50000, 700000), 2),
                    "hour": random.randint(8, 20),
                    "day_of_week": random.randint(0, 6),
                    "location": m["city"],
                    "device_id": f"DEV_{random.randint(1,10):04d}",
                    "is_new_device": 0, "transaction_count_1h": random.randint(1, 5),
                    "transaction_count_24h": random.randint(5, 25),
                    "amount_vs_avg_ratio": round(random.uniform(0.5, 2.0), 2),
                    "location_mismatch": 0
                }
            process_transaction(txn)
    st.success("Demo simulation complete! 10 transactions processed.")
    st.rerun()

# ─── TABS ───
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Analyze Transaction",
    "📊 Live Dashboard",
    "🤖 Agent Workflow",
    "📲 Notifications"
])

# ══════════════════════════════════════════════
# TAB 1: ANALYZE TRANSACTION
# ══════════════════════════════════════════════
with tab1:
    st.markdown("### Submit a Transaction for Analysis")
    st.markdown("Enter transaction details below. The full Agentic AI + ML pipeline will analyze it in real-time.")

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("**Transaction Details**")
        merchant_id = st.selectbox("Merchant", list(MERCHANTS.keys()),
                                    format_func=lambda x: f"{x} — {MERCHANTS[x]['name']}")
        amount = st.number_input("Amount (Rp)", min_value=1000.0, max_value=100000000.0,
                                  value=500000.0, step=10000.0, format="%.0f")
        hour = st.slider("Transaction Hour", 0, 23, 14)
        location = st.selectbox("Location", CITIES,
                                  index=CITIES.index(MERCHANTS[merchant_id]["city"]) if MERCHANTS[merchant_id]["city"] in CITIES else 0)

    with col_right:
        st.markdown("**Behavioral Context**")
        is_new_device = st.toggle("New/Unknown Device", value=False)
        location_mismatch = st.toggle("Location Mismatch", value=False)
        txn_count_1h = st.slider("Transactions in last 1 hour", 1, 30, 3)
        amount_ratio = st.slider("Amount vs. Average Ratio", 0.1, 25.0, 1.0, 0.1)

        st.markdown("**Presets:**")
        preset_cols = st.columns(3)
        if preset_cols[0].button("Normal Txn", use_container_width=True):
            st.session_state["preset"] = "normal"
        if preset_cols[1].button("Suspicious", use_container_width=True):
            st.session_state["preset"] = "suspicious"
        if preset_cols[2].button("High Fraud", use_container_width=True):
            st.session_state["preset"] = "fraud"

    st.markdown("---")
    analyze_btn = st.button("⚡ Analyze Transaction", use_container_width=True)

    if analyze_btn:
        txn_dict = {
            "transaction_id": f"TXN_{int(time.time()*1000)}",
            "merchant_id": merchant_id,
            "merchant_type": MERCHANTS[merchant_id]["type"],
            "amount": amount,
            "hour": hour,
            "day_of_week": datetime.now().weekday(),
            "location": location,
            "device_id": f"DEV_{random.randint(1,999):04d}",
            "is_new_device": int(is_new_device),
            "transaction_count_1h": txn_count_1h,
            "transaction_count_24h": txn_count_1h * 5,
            "amount_vs_avg_ratio": amount_ratio,
            "location_mismatch": int(location_mismatch)
        }

        with st.spinner("Running ML + Agent pipeline..."):
            case, ml_result = process_transaction(txn_dict)

        # ─── Results ───
        action = case["final_action"]
        score = ml_result["ensemble_score"]

        if action == "BLOCK":
            st.error(f"🚫 **TRANSACTION BLOCKED** — Risk Score: {score:.2%}")
        elif action == "STEP_UP_AUTH":
            st.warning(f"🔐 **STEP-UP AUTHENTICATION REQUIRED** — Risk Score: {score:.2%}")
        else:
            st.success(f"✅ **TRANSACTION APPROVED** — Risk Score: {score:.2%}")

        r1, r2, r3 = st.columns(3)
        r1.plotly_chart(make_gauge(ml_result["lgb_score"], "LightGBM Score"), use_container_width=True)
        r2.plotly_chart(make_gauge(ml_result["ae_score"], "Autoencoder Score"), use_container_width=True)
        r3.plotly_chart(make_gauge(score, "Ensemble Score"), use_container_width=True)

        st.markdown("**Agent Reasoning:**")
        st.info(case["resolution"]["reasoning"])

        st.markdown("**Notification sent to owner:**")
        st.markdown(f"""
        <div class='notification-card {case["notification"]["priority"]}'>
            <b>{case["notification"]["recipient"]}</b> &nbsp;·&nbsp;
            <span style='color:#888; font-size:12px'>{case["notification"]["timestamp"][:19]}</span><br>
            <p style='margin:8px 0 0; color:#ddd;'>{case["notification"]["message"]}</p>
        </div>
        """, unsafe_allow_html=True)

        if case["investigation"]["evidence_flags"]:
            st.markdown("**Evidence Flags:**")
            for flag in case["investigation"]["evidence_flags"]:
                st.markdown(f"- 🔴 `{flag}`")

# ══════════════════════════════════════════════
# TAB 2: LIVE DASHBOARD
# ══════════════════════════════════════════════
with tab2:
    if not st.session_state.cases:
        st.info("No transactions processed yet. Use the 'Analyze Transaction' tab or run the Demo Simulation from the sidebar.")
    else:
        cases = st.session_state.cases
        actions = [c["final_action"] for c in cases]
        scores = [c["ml_result"]["ensemble_score"] for c in cases]

        # KPI Row
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Transactions", len(cases))
        k2.metric("Blocked", actions.count("BLOCK"), delta=None)
        k3.metric("Step-Up Auth", actions.count("STEP_UP_AUTH"))
        k4.metric("Approved", actions.count("APPROVE"))
        k5.metric("Avg Risk Score", f"{sum(scores)/len(scores):.2%}")

        st.markdown("---")
        c1, c2 = st.columns(2)

        with c1:
            # Action distribution pie
            action_counts = {"BLOCK": actions.count("BLOCK"),
                             "STEP_UP_AUTH": actions.count("STEP_UP_AUTH"),
                             "APPROVE": actions.count("APPROVE")}
            fig_pie = go.Figure(go.Pie(
                labels=["Blocked", "Step-Up Auth", "Approved"],
                values=list(action_counts.values()),
                hole=0.5,
                marker_colors=["#E74C3C", "#E67E22", "#27AE60"],
                textfont={"color": "white"}
            ))
            fig_pie.update_layout(
                title="Transaction Decision Distribution",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#ccc"}, height=300,
                legend={"font": {"color": "#ccc"}},
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            # Risk score histogram
            fig_hist = go.Figure(go.Histogram(
                x=scores, nbinsx=20,
                marker_color="#4A90D9", opacity=0.8
            ))
            fig_hist.add_vline(x=0.45, line_dash="dash", line_color="#E67E22",
                               annotation_text="Step-Up", annotation_font_color="#E67E22")
            fig_hist.add_vline(x=0.75, line_dash="dash", line_color="#E74C3C",
                               annotation_text="Block", annotation_font_color="#E74C3C")
            fig_hist.update_layout(
                title="Risk Score Distribution",
                xaxis_title="Ensemble Risk Score",
                yaxis_title="Count",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#ccc"}, height=300,
                xaxis={"gridcolor": "#222"}, yaxis={"gridcolor": "#222"},
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # Transaction Table
        st.markdown("### Recent Transactions")
        table_data = []
        for c in reversed(cases[-20:]):
            table_data.append({
                "Transaction ID": c["transaction"]["transaction_id"],
                "Merchant": MERCHANTS.get(c["transaction"]["merchant_id"], {}).get("name", c["transaction"]["merchant_id"]),
                "Amount (Rp)": f"{c['transaction']['amount']:,.0f}",
                "Location": c["transaction"]["location"],
                "Risk Score": f"{c['ml_result']['ensemble_score']:.2%}",
                "Decision": c["final_action"],
                "Time (ms)": c["processing_time_ms"]
            })
        df = pd.DataFrame(table_data)

        def color_decision(val):
            if val == "BLOCK": return "background-color: #3d0a0a; color: #E74C3C"
            elif val == "STEP_UP_AUTH": return "background-color: #3d2a0a; color: #E67E22"
            else: return "background-color: #0a2a0a; color: #27AE60"

        styled = df.style.applymap(color_decision, subset=["Decision"])
        st.dataframe(styled, use_container_width=True, height=400)

# ══════════════════════════════════════════════
# TAB 3: AGENT WORKFLOW
# ══════════════════════════════════════════════
with tab3:
    st.markdown("### Multi-Agent System Workflow")
    st.markdown("Select a processed case to see the full agent reasoning chain.")

    if not st.session_state.cases:
        st.info("No cases yet. Analyze a transaction first.")
    else:
        case_ids = [c["case_id"] for c in reversed(st.session_state.cases)]
        selected_id = st.selectbox("Select Case", case_ids)
        selected_case = next(c for c in st.session_state.cases if c["case_id"] == selected_id)

        ml = selected_case["ml_result"]
        mon = selected_case["monitor_result"]
        inv = selected_case["investigation"]
        res = selected_case["resolution"]
        notif = selected_case["notification"]
        txn = selected_case["transaction"]

        # Header
        action = selected_case["final_action"]
        color = "#E74C3C" if action == "BLOCK" else ("#E67E22" if action == "STEP_UP_AUTH" else "#27AE60")
        st.markdown(f"""
        <div style='background:#1a1d2e; border-radius:12px; padding:16px; margin-bottom:16px;
                    border:1px solid {color};'>
            <span style='font-size:18px; font-weight:700; color:{color};'>{action}</span>
            &nbsp;&nbsp;
            <span style='color:#888; font-size:13px;'>Case: {selected_id}</span>
            &nbsp;·&nbsp;
            <span style='color:#888; font-size:13px;'>Rp {txn['amount']:,.0f} · {txn['location']} · {txn['hour']:02d}:00</span>
            &nbsp;·&nbsp;
            <span style='color:#888; font-size:13px;'>⏱ {selected_case['processing_time_ms']}ms</span>
        </div>
        """, unsafe_allow_html=True)

        # Agent Steps
        steps = [
            {
                "name": "1. ML Core Engine",
                "icon": "🧠",
                "color": "#27AE60",
                "content": f"""
                **LightGBM Score:** {ml['lgb_score']:.4f} &nbsp;|&nbsp;
                **Autoencoder Score:** {ml['ae_score']:.4f} &nbsp;|&nbsp;
                **Ensemble Score:** {ml['ensemble_score']:.4f}

                **Initial Decision:** `{ml['decision']}` &nbsp;|&nbsp; **Risk Level:** `{ml['risk_level']}`

                **Factors:** {ml['explanation']}
                """
            },
            {
                "name": "2. Monitor Agent",
                "icon": "👁",
                "color": "#4A90D9",
                "content": f"""
                **Action Triggered:** `{mon['action']}`

                **Reason:** {mon['trigger_reason']}
                """
            },
            {
                "name": "3. Investigator Agent",
                "icon": "🔍",
                "color": "#8E44AD",
                "content": f"""
                **Tools Called:** {', '.join([f'`{t}`' for t in inv['tools_called']])}

                **Device Check:** {'⚠️ Unknown device' if not inv['device_check']['is_known_device'] else '✅ Known device'}
                &nbsp;|&nbsp;
                **Location:** {'⚠️ Mismatch' if inv['location_check']['mismatch'] else '✅ Consistent'}

                **Fraud Patterns Matched:** {inv['pattern_match']['pattern_count']}
                {(' — ' + ', '.join(inv['pattern_match']['matched_patterns'][:2])) if inv['pattern_match']['matched_patterns'] else ''}

                **Evidence Flags:** {', '.join([f'`{f}`' for f in inv['evidence_flags']]) if inv['evidence_flags'] else 'None'}

                **Fraud Confidence:** {inv['fraud_confidence']:.0%}
                """
            },
            {
                "name": "4. Resolution Agent",
                "icon": "⚡",
                "color": "#E67E22",
                "content": f"""
                **Final Decision:** `{res['final_action']}`

                **Reasoning:**
                > {res['reasoning']}
                """
            },
            {
                "name": "5. Communicator Agent",
                "icon": "📢",
                "color": "#E74C3C" if notif['priority'] == "HIGH" else ("#E67E22" if notif['priority'] == "MEDIUM" else "#27AE60"),
                "content": f"""
                **Recipient:** {notif['recipient']} &nbsp;|&nbsp; **Priority:** `{notif['priority']}` &nbsp;|&nbsp; **Channel:** {notif['channel']}

                **Message sent:**
                > {notif['message']}

                **Action Buttons:** {' · '.join([f'[{b}]' for b in notif['action_buttons']])}
                """
            }
        ]

        for step in steps:
            with st.expander(f"{step['icon']} {step['name']}", expanded=True):
                st.markdown(step["content"])

# ══════════════════════════════════════════════
# TAB 4: NOTIFICATIONS
# ══════════════════════════════════════════════
with tab4:
    st.markdown("### Notification Center")
    st.markdown("All notifications sent to SME owners by the Communicator Agent.")

    if not st.session_state.notifications:
        st.info("No notifications yet.")
    else:
        filter_priority = st.multiselect(
            "Filter by Priority",
            ["HIGH", "MEDIUM", "LOW"],
            default=["HIGH", "MEDIUM", "LOW"]
        )

        filtered = [n for n in reversed(st.session_state.notifications)
                    if n.get("priority") in filter_priority]

        st.markdown(f"Showing **{len(filtered)}** notifications")

        for notif in filtered:
            priority = notif.get("priority", "LOW")
            icon = "🚨" if priority == "HIGH" else ("⚠️" if priority == "MEDIUM" else "✅")
            notif_type = notif.get("notification_type", "")

            st.markdown(f"""
            <div class='notification-card {priority}'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <span style='font-weight:600; color:#ddd;'>{icon} {notif.get("recipient", "Owner")}</span>
                    <span style='font-size:11px; color:#555;'>{notif.get("timestamp", "")[:19]}</span>
                </div>
                <div style='font-size:11px; color:#666; margin:4px 0;'>
                    {notif.get("merchant_id", "")} · {notif_type} · {notif.get("channel", "")}
                </div>
                <div style='color:#ccc; font-size:13px; margin-top:8px; line-height:1.5;'>
                    {notif.get("message", "")}
                </div>
                <div style='margin-top:8px;'>
                    {''.join([f"<span style='background:#1e2130; border:1px solid #333; border-radius:4px; padding:3px 8px; font-size:11px; margin-right:6px; color:#aaa;'>{b}</span>" for b in notif.get("action_buttons", [])])}
                </div>
            </div>
            """, unsafe_allow_html=True)
