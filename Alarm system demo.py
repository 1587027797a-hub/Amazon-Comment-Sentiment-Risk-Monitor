# Amazon Risk System | UI Restored & Pseudo-Real-Time Fixed Edition
import gradio as gr
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
from datetime import datetime, timedelta
import warnings
import plotly.graph_objects as go
import random

warnings.filterwarnings('ignore')

# ==================== 1. Core Business Configuration ====================
RESEARCH_DATA = {
    'amazon': {
        'lda_topics': {
            0: {'name': 'Quality Issues', 'keywords': ['return', 'product', 'use', 'work', 'time']},
            1: {'name': 'Sensory Experience', 'keywords': ['smell', 'skin', 'scent']},
            2: {'name': 'Refund & Fraud ⚠️', 'keywords': ['refund', 'money', 'scam', 'fake', 'fraud']},
            3: {'name': 'Appearance Mismatch', 'keywords': ['color', 'wig', 'shape']},
            4: {'name': 'Nail Product Issues', 'keywords': ['nails', 'polish', 'nail']},
            5: {'name': 'Logistics & Delivery', 'keywords': ['shipping', 'delay', 'arrived']}
        }
    }
}

RISK_DECISION_MATRIX = {
    'critical': {'name': '🔴 CRITICAL RISK', 'color': '#DC143C', 'sla': 'Within 2 Hours',
                 'action': 'Escalate to PR & Legal Departments immediately'},
    'high': {'name': '🟠 HIGH RISK', 'color': '#FF8C00', 'sla': 'Within 4 Hours',
             'action': 'Prioritize processing and offer full refund compensation'},
    'medium': {'name': '🟡 MEDIUM RISK', 'color': '#FFD700', 'sla': 'Within 12 Hours',
               'action': 'Standard after-sales workflow'},
    'low': {'name': '🟢 LOW RISK', 'color': '#32CD32', 'sla': 'Within 24 Hours',
             'action': 'Routine response'}
}

# ==================== 2. AI Model Loading ====================
print("⏳ Loading AI Core Engine...")
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# ==================== 3. Real-time Simulation Engine ====================
def get_live_manager_data():
    """Generates simulated real-time data fluctuations for demo purposes"""
    regions = ["US", "UK", "Germany", "Japan", "France"]
    pi_values = [0.45 + random.uniform(-0.08, 0.08) for _ in range(5)]
    geo_df = pd.DataFrame({"Region": regions, "PI_Index": pi_values})

    history_df = pd.DataFrame({
        "Time": [(datetime.now() - timedelta(minutes=i * 10)).strftime("%H:%M") for i in range(10)][::-1],
        "PI": [0.3 + random.uniform(0, 0.4) for _ in range(10)]
    })

    kw_data = [
        ["Scam", f"+{random.randint(200, 350)}%", "Critical"],
        ["Late Delivery", f"+{random.randint(40, 120)}%", "High"],
        ["Skin Burn", f"+{random.randint(70, 150)}%", "Critical"],
        ["Missing Item", f"+{random.randint(10, 50)}%", "Normal"]
    ]
    return geo_df, history_df, kw_data


# ==================== 4. Business Logic ====================
def process_review(text):
    if not text or len(text) < 5: return "Please enter review content...", "", "", None, {}

    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=1)
        neg_score = float(probs[0, 0].cpu().numpy())

    text_lower = text.lower()
    topic_id = 0
    if any(w in text_lower for w in ['scam', 'fake', 'fraud']):
        topic_id = 2
    elif any(w in text_lower for w in ['shipping', 'late', 'delay']):
        topic_id = 5
    topic = RESEARCH_DATA['amazon']['lda_topics'][topic_id]

    pi_score = (neg_score * 0.4 + (abs(neg_score - 0.5) * 2) * 0.6)
    level = 'low'
    if neg_score > 0.8 or topic_id == 2:
        level = 'critical'
    elif neg_score > 0.5:
        level = 'high'
    elif neg_score > 0.3:
        level = 'medium'

    risk = RISK_DECISION_MATRIX[level]

    draft = f"""Dear Valued Customer,

My name is the Senior Relations Manager at Amazon. We have flagged your feedback regarding "{topic['name']}" as a high-priority case.

[Immediate Resolution]
- Case Reference: AMZ-{datetime.now().strftime('%m%d%f')[:8]}
- Action Plan: {risk['action']}

We maintain a zero-tolerance policy for quality and trust issues. Our team will follow up with you within {risk['sla']}.

Best regards,
Amazon Senior Support Team"""

    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=pi_score,
        gauge={'axis': {'range': [0, 1]}, 'bar': {'color': risk['color']}},
        title={'text': "Polarization Index"}
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))

    alert_html = f"""
    <div style="background-color: {risk['color']}15; border-left: 10px solid {risk['color']}; padding: 20px; border-radius: 8px;">
        <h2 style="color: {risk['color']}; margin: 0;">{risk['name']}</h2>
        <p style="margin: 10px 0;"><b>Identified Topic:</b> {topic['name']}</p>
        <p style="margin: 0;"><b>Response SLA:</b> <span style="color:red; font-weight:bold;">{risk['sla']}</span></p>
    </div>
    """

    action_md = f"### 🛡️ Expert Action Items\n- **Primary Instruction**: {risk['action']}\n- **Operational Note**: It is recommended to verify the SKU batch inventory for potential systemic defects."

    return alert_html, action_md, draft, fig, {"Sentiment": neg_score, "PI": pi_score}


# ==================== 5. UI Layout ====================
with gr.Blocks(theme=gr.themes.Soft(),
               css=".header {background:#232f3e; color:white; padding:15px; border-radius:8px;}") as demo:
    gr.HTML(
        "<div class='header'><h1>🛡️ Amazon After-Sales Risk Management & Sentiment Monitoring</h1><p>2026 Presentation Edition | Real-time Dynamics Monitoring Enabled</p></div>")

    timer = gr.Timer(4)  # Global timer for 4-second refresh

    with gr.Tabs():
        # --- TAB 1: Risk ASSESSMENT (Agent View) ---
        with gr.Tab("💬 Risk Assessment (Agent View)"):
            with gr.Row():
                # Left Side: Input Area
                with gr.Column(scale=1):
                    gr.Markdown("### 📥 Review Input Area")
                    input_text = gr.Textbox(label="Enter Customer Review Content", lines=10, placeholder="Paste customer feedback here...")

                    gr.Markdown("#### 💡 Quick Demo Samples")
                    with gr.Row():
                        ex1 = gr.Button("🟢 Logistics Delay", size="sm")
                        ex2 = gr.Button("🔴 Scam / Counterfeit", size="sm")

                    btn = gr.Button("🔍 Run Deep Risk Assessment", variant="primary", size="lg")

                    with gr.Accordion("📊 Analyst Metadata", open=False):
                        out_raw = gr.JSON()

                # Right Side: Analysis & Results
                with gr.Column(scale=1):
                    gr.Markdown("### 🚨 Evaluation & Decision Results")
                    out_plot = gr.Plot(label="Risk Gauge")  # Gauge at the top
                    out_alert = gr.HTML(
                        "<div style='color:gray; padding:20px; border:1px dashed #ccc'>Waiting for data analysis...</div>")
                    out_action = gr.Markdown("")

                    gr.Markdown("### 📧 AI-Enhanced Response Draft")
                    out_draft = gr.Textbox(label="Auto-generated Response", lines=10)
                    with gr.Row():
                        gr.Button("📋 Copy Draft", variant="secondary")
                        gr.Button("🛑 Emergency Escalation", variant="stop")

        # --- TAB 2: Global Trends (Manager View) ---
        with gr.Tab("📈 Global Crisis Trends (Manager View)"):
            gr.Markdown("## 🌐 Global Site Sentiment Overview (Auto-refreshes every 4s)")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📍 Regional Pressure Index")
                    live_geo_plot = gr.BarPlot(x="Region", y="PI_Index", title="Real-time Pressure by Region", y_lim=[0, 1])
                with gr.Column():
                    gr.Markdown("### 📈 Global Polarization Trend")
                    live_history_plot = gr.LinePlot(x="Time", y="PI", title="24-Hour Sample Trend")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### ⚡ Live High-Risk Keyword Monitoring")
                    live_kw_table = gr.DataFrame(headers=["Keyword", "Trend Growth", "Status"])
                with gr.Column():
                    gr.Markdown("""
                    ### 🔔 Intelligent Alert Logs
                    - **[System]** Polarization algorithm running in real-time...
                    - **[Alert]** Keyword 'Scam' frequency is rising in the US region.
                    - **[Automation]** Risk levels are being automatically mapped to high-priority queues.
                    """)

    # --- Interaction Logic ---
    # Sample Click Handlers
    ex1.click(lambda: "My package was delayed for 5 days. Very disappointed with the shipping service.", outputs=input_text)
    ex2.click(lambda: "TOTAL SCAM! This item is a cheap fake. I want my money back immediately!", outputs=input_text)

    # Process Button Handler
    btn.click(process_review, inputs=input_text, outputs=[out_alert, out_action, out_draft, out_plot, out_raw])

    # Pseudo-real-time Refresh Logic
    timer.tick(
        fn=get_live_manager_data,
        outputs=[live_geo_plot, live_history_plot, live_kw_table]
    )

    # Pre-load on startup
    demo.load(
        fn=get_live_manager_data,
        outputs=[live_geo_plot, live_history_plot, live_kw_table]
    )

if __name__ == "__main__":
    demo.launch()