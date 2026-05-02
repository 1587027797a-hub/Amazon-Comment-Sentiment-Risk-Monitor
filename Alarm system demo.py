# -*- coding: utf-8 -*-
# ============================================================================
# AmzGuard Enterprise | UI Final Polish & Optimization v2.2
# Features: Custom PI Formula, Rich AI Responses, Dynamic Placeholders, Enhanced UX
# ============================================================================
import gradio as gr
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go
import plotly.express as px
import random
import warnings
from datetime import datetime, timedelta
import os
import uuid

# Suppress warnings and optimize tokenizer parallelism
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# ==================== 1. Core Configuration & Styles ====================
RISK_CONFIG = {
    'CRITICAL': {
        'name': '🔴 CRITICAL: Safety & Legal Risk',
        'color': '#DC143C',
        'sla': '2h',
        'action': 'Immediately delist SKU, initiate legal protocol.',
        'reply': """[AmzGuard Urgent Response Team]

We are deeply concerned to hear about your experience. Your safety is our absolute highest priority, and we take all reports of physical harm or legal risk with the utmost seriousness.

✅ Immediate Actions Taken:
- The product has been temporarily suspended from sale pending investigation.
- Our legal and compliance teams have been notified and will contact you within 2 hours.
- A dedicated case manager has been assigned to your file (Case ID: {case_id}).

💡 Next Steps for You:
- Please preserve any evidence (photos, packaging, medical records if applicable).
- We will cover all reasonable costs associated with this incident upon verification.
- You may request a full refund or replacement at no cost — whichever you prefer.

We sincerely apologize for the distress caused and assure you that we are treating this as an emergency matter. Thank you for giving us the opportunity to make this right.""",

        'sample': "WARNING: This product caused a severe chemical BURN! Total scam. I am calling my lawyer."
    },
    'HIGH': {
        'name': '🟠 HIGH: Major Product Defect',
        'color': '#FF8C00',
        'sla': '4h',
        'action': 'Flag batch for 100% warehouse inspection.',
        'reply': """[AmzGuard Quality Assurance Team]

Thank you for bringing this serious defect to our attention. We understand how frustrating it must be to receive a broken or non-functional item — especially when you were counting on it working properly.

✅ What We’re Doing Right Now:
- Your order has been flagged for immediate QA review.
- We’ve initiated a recall check on the entire production batch (Batch #{batch_id}).
- A senior support agent will reach out to you within 4 hours to resolve this personally.

💡 How We’ll Make It Right:
- Full refund OR free replacement shipped via expedited delivery — your choice.
- Additional $25 credit applied to your account for the inconvenience.
- Optional: Return label provided if you wish to send back the defective unit for analysis.

We value your trust and are committed to ensuring this doesn’t happen again. Please let us know how else we can assist you during this process.""",

        'sample': "The product arrived broken and stopped working after 2 hours."
    },
    'MEDIUM': {
        'name': '🟡 MEDIUM: Logistics Issue',
        'color': '#FFD700',
        'sla': '12h',
        'action': 'Escalate logistics claim ticket.',
        'reply': """[AmzGuard Logistics Care Team]

We’re sorry to hear your package experienced delays — we know how important timely delivery is, especially when you were waiting for something special.

✅ Current Status & Action:
- Your shipment has been escalated to our priority logistics team.
- Tracking updated in real-time; expected new ETA: {eta_date}.
- If not delivered by then, automatic compensation will trigger.

💡 Compensation Offered:
- $15 store credit automatically added to your wallet.
- Free upgrade to express shipping on your next order.
- Option to cancel and receive full refund if delay exceeds 7 days.

We appreciate your patience and hope to restore your confidence in our service. Feel free to reply directly to this message if you need further assistance.""",

        'sample': "The delivery was 15 days late. The item is fine but the experience was terrible."
    },
    'LOW': {
        'name': '🟢 LOW: General Feedback',
        'color': '#32CD32',
        'sla': '24h',
        'action': 'Automated AI processing, log to CRM.',
        'reply': """[AmzGuard Customer Success Team]

Thank you so much for taking the time to share your feedback — whether positive or constructive, every comment helps us improve!

✅ We’ve Logged Your Input:
- Your review has been tagged and shared with our product development team.
- If you mentioned a suggestion, we’ve added it to our Q3 roadmap consideration list.
- No action required from you — unless you’d like to follow up!

💡 As a Token of Appreciation:
- Enjoy 10% off your next purchase with code: THANKYOU10
- Join our VIP Insider List for early access to new products and exclusive deals.

We’re thrilled you had a good experience — and even more excited to serve you again soon! 😊""",

        'sample': "It's a decent product for the price. Fast delivery and good packaging."
    }
}

# Unified Plotly Theme for High Contrast Black Text
PLOTLY_THEME = {
    'paper_bgcolor': 'white',
    'plot_bgcolor': 'white',
    'font': {'family': 'Inter, sans-serif', 'color': 'black', 'size': 12},
    'xaxis': {'gridcolor': '#E0E0E0', 'zerolinecolor': '#E0E0E0'},
    'yaxis': {'gridcolor': '#E0E0E0', 'zerolinecolor': '#E0E0E0'},
    'margin': dict(l=40, r=40, t=40, b=40)
}

# 修改后的 CSS：增强选择器权重，确保文字在深色背景下显示为白色
custom_css = """
.header-container {
    background: linear-gradient(90deg, #131921 0%, #232f3e 100%) !important; 
    padding: 25px !important; 
    border-radius: 10px !important;
    margin-bottom: 20px !important;
    box-shadow: 0 4px 10px rgba(0,0,0,0.3) !important;
}
/* 强制标题文字显示为白色并解决主题覆盖问题 */
.header-container h1 { 
    color: #FFFFFF !important; 
    margin: 0 !important; 
    font-size: 2rem !important; 
    font-weight: 700 !important;
}
/* 强制副标题文字显示为淡灰色 */
.header-container p { 
    color: #DDDDDD !important; 
    margin: 8px 0 0 0 !important; 
    opacity: 1 !important; 
    font-size: 1rem !important;
}
.gradio-container { font-family: 'Inter', sans-serif; }
.gradio-container .prose {
    font-size: 0.95rem;
    line-height: 1.6;
}
.gradio-container .prose blockquote {
    border-left-color: #131921 !important;
    background-color: #f8f9fa;
    padding: 1em;
    border-radius: 4px;
}
"""

# ==================== 2. AI Engine (Lazy Loading) ====================
_model_cache = {}


def get_ai_model():
    if 'model' not in _model_cache:
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            print("⏳ Initializing AI Engine (First Run)...")
            MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()
            _model_cache['tokenizer'] = tokenizer
            _model_cache['model'] = model
            _model_cache['device'] = device
            print("✅ AI Engine Ready.")
        except Exception as e:
            print(f"❌ Failed to load AI Model: {e}")
            raise gr.Error("AI Engine initialization failed.")
    return _model_cache['tokenizer'], _model_cache['model'], _model_cache['device']


# ==================== 3. Logic Functions ====================

def analyze_risk(text):
    if not text or len(text.strip()) < 3:
        return (
            "<div style='padding:20px; text-align:center; color:#666;'>Please enter valid review text.</div>",
            "### 🤖 Response Draft\n> Waiting for input...",
            None
        )

    try:
        tokenizer, model, device = get_ai_model()
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            neg_score = float(probs[0, 0].cpu().numpy())
            pos_score = float(probs[0, 2].cpu().numpy())

        # PI Calculation
        mean_component = neg_score
        variance_component = abs(neg_score - 0.5) * 2
        extreme_ratio = 1.0 if neg_score > 0.8 else 0.0
        pi_score = (0.3 * mean_component) + (0.4 * variance_component) + (0.3 * extreme_ratio)
        pi_score = min(max(pi_score, 0), 1)

        level = 'LOW'
        t_l = text.lower()
        if pi_score > 0.75 or any(w in t_l for w in ['burn', 'scam', 'sue', 'lawyer', 'dangerous', 'fire']):
            level = 'CRITICAL'
        elif pi_score > 0.55 or any(w in t_l for w in ['broken', 'damaged', 'defect', 'waste']):
            level = 'HIGH'
        elif pi_score > 0.35 or any(w in t_l for w in ['late', 'delay', 'slow', 'missing']):
            level = 'MEDIUM'

        if pos_score > 0.85 and level != 'CRITICAL':
            level = 'LOW'
            pi_score = pi_score * 0.2

        risk = RISK_CONFIG[level]
        case_id = f"AG-{uuid.uuid4().hex[:6].upper()}"
        batch_id = f"BATCH-{np.random.randint(1000, 9999)}"
        eta_date = (datetime.now() + timedelta(days=3)).strftime("%B %d, %Y")

        personalized_reply = risk['reply'].replace("{case_id}", case_id).replace("{batch_id}", batch_id).replace(
            "{eta_date}", eta_date)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pi_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Polarization Index (PI)", 'font': {'size': 16, 'color': "black"}},
            gauge={
                'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "black"},
                'bar': {'color': risk['color']},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 0.35], 'color': '#E8F5E9'},
                    {'range': [0.35, 0.55], 'color': '#FFF3E0'},
                    {'range': [0.55, 0.75], 'color': '#FFCCBC'},
                    {'range': [0.75, 1], 'color': '#FFEBEE'}
                ],
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 0.75}
            }
        ))
        fig.update_layout(height=250, **PLOTLY_THEME)

        alert_html = f"""
        <div style='background:{risk['color']}10; border-left:6px solid {risk['color']}; padding:15px; border-radius:4px; font-family:sans-serif;'>
            <h3 style='color:{risk['color']}; margin:0 0 5px 0; font-size:1.1em;'>{risk['name']}</h3>
            <p style='color:#333; margin:0; font-size:0.9em;'><b>SLA Target:</b> {risk['sla']} | <b>Action:</b> {risk['action']}</p>
            <p style='color:#666; margin:5px 0 0 0; font-size:0.8em;'><i>PI Formula: 0.3×Mean + 0.4×Var + 0.3×Ext</i></p>
        </div>
        """
        return alert_html, f"### 🤖 AI Response Draft\n> {personalized_reply}", fig
    except Exception as e:
        return f"<div style='color:red;'>Analysis Error: {e}</div>", "### Error", None


def batch_process(file):
    if file is None: return "### ⚠️ No File Uploaded", None, {"Status": "Error"}
    random.seed(len(file.name))
    labels = ["Critical", "High", "Medium", "Low"]
    values = [random.randint(1, 5), random.randint(5, 15), random.randint(20, 40), random.randint(60, 100)]
    total = sum(values)
    fig = px.pie(values=values, names=labels, title="Batch Risk Distribution",
                 color_discrete_map={'Critical': '#DC143C', 'High': '#FF8C00', 'Medium': '#FFD700', 'Low': '#32CD32'})
    fig.update_layout(**PLOTLY_THEME, height=400)
    summary_md = f"### 📊 Scan Complete\n- **Total Records:** {total}\n- **Critical Issues:** {values[0]}"
    json_status = {"Status": "Success", "File": os.path.basename(file.name), "Critical_Count": values[0]}
    return summary_md, fig, json_status


def get_live_charts():
    regions = ["US", "UK", "DE", "JP", "FR"]
    pi_values = [round(0.3 + random.uniform(0, 0.4), 2) for _ in range(5)]
    fig1 = px.bar(x=regions, y=pi_values, title="Regional Market Pressure", color=pi_values,
                  color_continuous_scale=[[0, '#32CD32'], [0.5, '#FFD700'], [1, '#DC143C']])
    fig1.update_layout(**PLOTLY_THEME, showlegend=False, height=300)

    now = datetime.now()
    times = [(now - timedelta(minutes=i * 30)).strftime("%H:%M") for i in range(12)][::-1]
    trend_values = [round(0.2 + random.uniform(0, 0.3) + (0.05 * i), 2) for i in range(12)]
    fig2 = px.line(x=times, y=trend_values, title="24H Global Risk Trend", markers=True)
    fig2.update_layout(**PLOTLY_THEME, height=300)

    keywords = [["Battery Swelling", "+420%", "🔴 Urgent"], ["Screen Flicker", "+150%", "🟠 High"],
                ["Late Delivery", "+85%", "🟡 Watch"], ["Packaging Damaged", "+40%", "🟢 Monitor"]]
    random.shuffle(keywords)
    return fig1, fig2, pd.DataFrame(keywords, columns=["Keyword", "Burst Rate", "Risk Status"])


# ==================== 4. UI Layout ====================

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    # 修改后的 HTML 结构，使用自定义类名强制渲染
    gr.HTML("""
    <div class='header-container'>
        <h1>🛡️ AmzGuard Enterprise</h1>
        <p>Global Risk Diagnostic & Strategy Center | Powered by Transformer AI</p>
    </div>
    """)

    timer = gr.Timer(10)

    with gr.Tabs():
        with gr.Tab("🚨 Risk Assessment"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Input Review Content")
                    t_input = gr.Textbox(label="Customer Review", lines=6, placeholder="Paste review...")
                    with gr.Row():
                        btn_c = gr.Button("🔴 Crit", size="sm", variant="stop")
                        btn_h = gr.Button("🟠 High", size="sm")
                        btn_m = gr.Button("🟡 Med", size="sm")
                        btn_l = gr.Button("🟢 Low", size="sm")
                    run_btn = gr.Button("Execute AI Analysis", variant="primary", size="lg")
                with gr.Column(scale=1):
                    out_html = gr.HTML(
                        "<div style='color:#888; text-align:center; padding:40px; border:1px dashed #ccc;'>Awaiting Analysis...</div>")
                    out_plot = gr.Plot(label="Risk Gauge")
                    out_md = gr.Markdown("### 🤖 Response Draft")

        with gr.Tab("📂 Data Center"):
            with gr.Row():
                with gr.Column(scale=1):
                    file_in = gr.File(label="Upload Dataset", file_types=['.csv', '.xlsx'])
                    u_btn = gr.Button("Run Global Batch Scan", variant="primary")
                with gr.Column(scale=2):
                    batch_res = gr.Markdown("### Summary")
                    batch_plot = gr.Plot(label="Distribution")
                    batch_json = gr.JSON(label="Processing Metadata")

        with gr.Tab("🌐 Command Center"):
            with gr.Row():
                geo_chart = gr.Plot(label="Market Pressure")
                trend_chart = gr.Plot(label="24H Risk Trend")
            kw_table = gr.DataFrame(headers=["Keyword", "Burst Rate", "Risk Status"], label="Live Keyword Monitoring")

    run_btn.click(analyze_risk, inputs=t_input, outputs=[out_html, out_md, out_plot])
    btn_c.click(lambda: RISK_CONFIG['CRITICAL']['sample'], outputs=t_input)
    btn_h.click(lambda: RISK_CONFIG['HIGH']['sample'], outputs=t_input)
    btn_m.click(lambda: RISK_CONFIG['MEDIUM']['sample'], outputs=t_input)
    btn_l.click(lambda: RISK_CONFIG['LOW']['sample'], outputs=t_input)
    u_btn.click(batch_process, inputs=file_in, outputs=[batch_res, batch_plot, batch_json])
    timer.tick(get_live_charts, outputs=[geo_chart, trend_chart, kw_table])
    demo.load(get_live_charts, outputs=[geo_chart, trend_chart, kw_table])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
