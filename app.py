import streamlit as st
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.set_page_config(
    page_title="MediPredict AI | Vishal Jha",
    page_icon="🧬",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&family=Share+Tech+Mono&display=swap');

* { margin: 0; padding: 0; box-sizing: border-box; }

:root {
    --neon-cyan: #00f5ff;
    --neon-green: #00ff88;
    --neon-pink: #ff006e;
    --neon-purple: #7b2fff;
    --dark-bg: #020c1b;
    --card-bg: rgba(0, 245, 255, 0.04);
    --border: rgba(0, 245, 255, 0.2);
    --text-main: #ccd6f6;
    --text-dim: #8892b0;
}

.stApp {
    background: #020c1b !important;
    font-family: 'Rajdhani', sans-serif !important;
}

.stApp::before {
    content: '';
    position: fixed; inset: 0; z-index: 0;
    background:
        radial-gradient(ellipse at 20% 50%, rgba(123,47,255,0.12) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 20%, rgba(0,245,255,0.10) 0%, transparent 55%),
        radial-gradient(ellipse at 60% 80%, rgba(0,255,136,0.07) 0%, transparent 50%);
    pointer-events: none;
}

.hero {
    text-align: center;
    padding: 3rem 1rem 2rem;
}
.hero-tag {
    display: inline-block;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: #00f5ff;
    letter-spacing: 4px;
    text-transform: uppercase;
    border: 1px solid rgba(0,245,255,0.2);
    padding: 4px 16px;
    border-radius: 2px;
    margin-bottom: 1.2rem;
    animation: fadeSlideDown 0.8s ease both;
}
.hero-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 3.5rem;
    font-weight: 900;
    background: linear-gradient(135deg, #00f5ff 0%, #00ff88 50%, #00f5ff 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 3s linear infinite, fadeSlideDown 0.8s 0.2s ease both;
    margin-bottom: 0.5rem;
}
.hero-subtitle {
    font-size: 1.1rem;
    color: #8892b0;
    letter-spacing: 2px;
    font-weight: 300;
    animation: fadeSlideDown 0.8s 0.4s ease both;
    margin-bottom: 0.4rem;
}
.hero-author {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.95rem;
    color: #ff006e;
    letter-spacing: 3px;
    animation: fadeSlideDown 0.8s 0.6s ease both;
}
.divider {
    width: 120px; height: 2px;
    background: linear-gradient(90deg, transparent, #00f5ff, transparent);
    margin: 1.5rem auto;
}

.stats-row {
    display: flex;
    justify-content: center;
    gap: 1.5rem;
    flex-wrap: wrap;
    margin: 0 auto 2rem;
    max-width: 900px;
}
.stat-card {
    background: rgba(0,245,255,0.04);
    border: 1px solid rgba(0,245,255,0.2);
    border-radius: 8px;
    padding: 1rem 2rem;
    text-align: center;
    min-width: 150px;
    transition: transform 0.3s, box-shadow 0.3s;
    position: relative;
    overflow: hidden;
}
.stat-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00f5ff, #7b2fff);
}
.stat-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 30px rgba(0,245,255,0.15);
}
.stat-value {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #00f5ff;
    display: block;
}
.stat-label {
    font-size: 0.75rem;
    color: #8892b0;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 2px;
}

.section-heading {
    font-family: 'Orbitron', sans-serif;
    font-size: 1rem;
    color: #00f5ff;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 12px;
}
.section-heading::after {
    content: '';
    flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(0,245,255,0.2), transparent);
}

.glass-panel {
    background: rgba(0,245,255,0.04);
    border: 1px solid rgba(0,245,255,0.2);
    border-radius: 12px;
    padding: 1.8rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.glass-panel::before {
    content: '';
    position: absolute; top: 0; left: 0;
    width: 60px; height: 60px;
    background: linear-gradient(135deg, rgba(0,245,255,0.1), transparent);
    border-radius: 0 0 60px 0;
}

div[data-testid="stNumberInput"] label {
    color: #ccd6f6 !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
}
div[data-testid="stNumberInput"] input {
    background: rgba(0,245,255,0.05) !important;
    border: 1px solid rgba(0,245,255,0.2) !important;
    border-radius: 6px !important;
    color: #00f5ff !important;
    font-family: 'Share Tech Mono', monospace !important;
}

div[data-testid="stButton"] button {
    background: linear-gradient(135deg, rgba(0,245,255,0.1), rgba(123,47,255,0.2)) !important;
    border: 2px solid #00f5ff !important;
    color: #00f5ff !important;
    font-family: 'Orbitron', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    letter-spacing: 4px !important;
    padding: 0.8rem 2rem !important;
    border-radius: 6px !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase !important;
}
div[data-testid="stButton"] button:hover {
    background: linear-gradient(135deg, rgba(0,245,255,0.25), rgba(123,47,255,0.35)) !important;
    box-shadow: 0 0 30px rgba(0,245,255,0.4) !important;
    transform: translateY(-2px) !important;
}

.result-danger {
    background: linear-gradient(135deg, rgba(255,0,110,0.12), rgba(255,0,110,0.05));
    border: 1px solid rgba(255,0,110,0.5);
    border-left: 4px solid #ff006e;
    border-radius: 8px;
    padding: 1.5rem;
    margin: 1rem 0;
}
.result-safe {
    background: linear-gradient(135deg, rgba(0,255,136,0.12), rgba(0,255,136,0.05));
    border: 1px solid rgba(0,255,136,0.5);
    border-left: 4px solid #00ff88;
    border-radius: 8px;
    padding: 1.5rem;
    margin: 1rem 0;
}
.result-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
.result-danger .result-title { color: #ff006e; }
.result-safe .result-title   { color: #00ff88; }
.result-desc {
    color: #8892b0;
    font-size: 1rem;
    line-height: 1.6;
}

.conf-wrap { margin: 1.2rem 0; }
.conf-label {
    display: flex;
    justify-content: space-between;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.8rem;
    color: #8892b0;
    margin-bottom: 6px;
}
.conf-track {
    height: 8px;
    background: rgba(255,255,255,0.05);
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 10px;
}
.conf-fill-safe {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #00ff88, #00c9ff);
}
.conf-fill-danger {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #ff006e, #ff8c00);
}

.data-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.85rem;
    margin-top: 1rem;
}
.data-table th {
    color: #00f5ff;
    text-align: left;
    padding: 8px 12px;
    border-bottom: 1px solid rgba(0,245,255,0.2);
    letter-spacing: 2px;
    font-size: 0.75rem;
    text-transform: uppercase;
}
.data-table td {
    color: #ccd6f6;
    padding: 8px 12px;
    border-bottom: 1px solid rgba(0,245,255,0.05);
}
.data-table tr:hover td {
    background: rgba(0,245,255,0.04);
    color: #00f5ff;
}

.footer {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    border-top: 1px solid rgba(0,245,255,0.2);
    margin-top: 3rem;
}
.footer-name {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.4rem;
    font-weight: 900;
    background: linear-gradient(90deg, #00f5ff, #7b2fff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 4px;
    margin-bottom: 0.4rem;
}
.footer-sub {
    font-size: 0.8rem;
    color: #8892b0;
    letter-spacing: 2px;
    font-family: 'Share Tech Mono', monospace;
}

@keyframes shimmer {
    0%   { background-position: 0% center; }
    100% { background-position: 200% center; }
}
@keyframes fadeSlideDown {
    from { opacity: 0; transform: translateY(-20px); }
    to   { opacity: 1; transform: translateY(0); }
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; }
</style>
""", unsafe_allow_html=True)

# HERO
st.markdown("""
<div class="hero">
    <div class="hero-tag">🧬 AI-POWERED MEDICAL DIAGNOSTICS</div>
    <div class="hero-title">MEDIPREDICT AI</div>
    <div class="hero-subtitle">Advanced Diabetes Risk Analysis System</div>
    <div class="hero-author">// Developed by VISHAL JHA //</div>
    <div class="divider"></div>
</div>
<div class="stats-row">
    <div class="stat-card">
        <span class="stat-value">81%</span>
        <span class="stat-label">Accuracy</span>
    </div>
    <div class="stat-card">
        <span class="stat-value">3</span>
        <span class="stat-label">ML Models</span>
    </div>
    <div class="stat-card">
        <span class="stat-value">8</span>
        <span class="stat-label">Parameters</span>
    </div>
    <div class="stat-card">
        <span class="stat-value">AI</span>
        <span class="stat-label">Powered</span>
    </div>
</div>
""", unsafe_allow_html=True)

# INPUT
st.markdown('<div class="glass-panel"><div class="section-heading">📋 Patient Data Input</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
with col2:
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
with col3:
    insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
with col4:
    dpf = st.number_input("Diabetes Pedigree", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
    age = st.number_input("Age", min_value=1, max_value=120, value=30)

st.markdown('</div>', unsafe_allow_html=True)

predict = st.button("⚡  RUN DIAGNOSTIC ANALYSIS", use_container_width=True)

if predict:
    input_data = np.array([[pregnancies, glucose, blood_pressure,
                             skin_thickness, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0]
    safe_pct = probability[0] * 100
    danger_pct = probability[1] * 100

    st.markdown('<div class="glass-panel"><div class="section-heading">📊 Diagnostic Result</div>', unsafe_allow_html=True)

    if prediction[0] == 1:
        st.markdown(f"""
        <div class="result-danger">
            <div class="result-title">⚠️ HIGH RISK — DIABETES DETECTED</div>
            <div class="result-desc">
                AI model ne elevated risk indicators detect kiye hain.<br>
                Certified physician se turant milne ki salah di jaati hai.
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-safe">
            <div class="result-title">✅ LOW RISK — NO DIABETES DETECTED</div>
            <div class="result-desc">
                Analysis mein diabetes ki low probability hai is waqt.<br>
                Healthy lifestyle maintain karo aur regular check-up karwao.
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="conf-wrap">
        <div class="conf-label"><span>NO DIABETES</span><span>{safe_pct:.1f}%</span></div>
        <div class="conf-track"><div class="conf-fill-safe" style="width:{safe_pct}%"></div></div>
        <div class="conf-label"><span>DIABETES RISK</span><span>{danger_pct:.1f}%</span></div>
        <div class="conf-track"><div class="conf-fill-danger" style="width:{danger_pct}%"></div></div>
    </div>
    <table class="data-table">
        <tr><th>Parameter</th><th>Value</th><th>Unit</th></tr>
        <tr><td>Pregnancies</td><td>{pregnancies}</td><td>count</td></tr>
        <tr><td>Glucose Level</td><td>{glucose}</td><td>mg/dL</td></tr>
        <tr><td>Blood Pressure</td><td>{blood_pressure}</td><td>mm Hg</td></tr>
        <tr><td>Skin Thickness</td><td>{skin_thickness}</td><td>mm</td></tr>
        <tr><td>Insulin Level</td><td>{insulin}</td><td>μU/mL</td></tr>
        <tr><td>BMI</td><td>{bmi}</td><td>kg/m²</td></tr>
        <tr><td>Diabetes Pedigree</td><td>{dpf}</td><td>score</td></tr>
        <tr><td>Age</td><td>{age}</td><td>years</td></tr>
    </table>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# FOOTER
st.markdown("""
<div class="footer">
    <div class="footer-name">VISHAL JHA</div>
    <div class="footer-sub">Final Year Project &nbsp;|&nbsp; Computer Science &nbsp;|&nbsp; AI / Machine Learning &nbsp;|&nbsp; 2026</div>
    <br>
    <div class="footer-sub" style="color:rgba(255,255,255,0.2); font-size:0.7rem;">
        ⚠️ For educational purposes only. Not a substitute for professional medical advice.
    </div>
</div>
""", unsafe_allow_html=True)