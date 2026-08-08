import streamlit as st
import time
import json
import numpy as np
from multimodal_orchestrator import OmniFusionOrchestrator

# Page Configuration
st.set_page_config(
    page_title="OmniFusion-AI | Multimodal Fusion & Cross-Attention Intelligence",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling & CSS
st.markdown("""
<style>
    .stApp {
        background-color: #070a14;
        color: #f8fafc;
    }
    .main-header {
        font-family: 'Outfit', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #06b6d4, #3b82f6, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .card-box {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.35);
    }
    .badge {
        padding: 0.25rem 0.65rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    .badge-cyan { background: rgba(6, 182, 212, 0.2); color: #06b6d4; }
    .badge-pink { background: rgba(236, 72, 153, 0.2); color: #ec4899; }
    .badge-green { background: rgba(16, 185, 129, 0.2); color: #10b981; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_orchestrator():
    return OmniFusionOrchestrator()

orchestrator = get_orchestrator()

# Sidebar Navigation & Domain Presets
st.sidebar.title("🌌 OmniFusion-AI")
st.sidebar.caption("Multimodal Fusion & Cross-Attention Intelligence")

domain_choice = st.sidebar.selectbox(
    "Select Multimodal Application Domain",
    [
        "🏥 Medical Diagnostics (X-ray + Clinical Notes + ECG)",
        "🚗 Autonomous Vehicles (Camera + LiDAR + RADAR)",
        "🛍️ E-Commerce Search (Image + Text Modifier)",
        "🔬 Custom Multimodal Playground"
    ]
)

# Set Default Inputs based on Preset
if "Medical" in domain_choice:
    default_text = "Patient presenting with acute shortness of breath, bilateral lower lobe wheezing, and fever for 4 days."
    default_vision = "Chest X-Ray radiograph demonstrating mild patchy opacity in right lower lobe."
    default_sensor = "ECG sound wave recording: Sinus tachycardia at 104 bpm, ST elevation unremarkable."
    domain_name = "Medical Diagnostics"
elif "Autonomous" in domain_choice:
    default_text = "Navigate 4-way intersection in rain with pedestrian standing near crosswalk."
    default_vision = "Front camera RGB footage showing wet asphalt and low-contrast pedestrian silhouette at 18 meters."
    default_sensor = "LiDAR 3D point cloud cluster detected at [x: 3.2, y: 18.1, z: 0.0] with RADAR velocity -1.2 m/s."
    domain_name = "Autonomous Driving"
elif "E-Commerce" in domain_choice:
    default_text = "Find high-top athletic running shoes like this image, but in vibrant crimson red."
    default_vision = "Product Image: High-top white and navy cushioned running sneaker."
    default_sensor = "Customer filter sensor: Men's Size 10, Carbon-fiber plate enabled."
    domain_name = "E-Commerce Multimodal Search"
else:
    default_text = "Describe the relationship between visual geometry and structural stability."
    default_vision = "Image showing triangular truss bridge superstructure."
    default_sensor = "Acoustic sensor telemetry: Low structural vibration frequency (12 Hz)."
    domain_name = "Custom Multimodal"

# Main Layout
st.markdown('<div class="main-header">🌌 OmniFusion-AI Multimodal Intelligence</div>', unsafe_allow_html=True)
st.caption("Early Fusion (Data-Level) • Late Fusion (Decision-Level) • Intermediate Cross-Attention • CLIP Contrastive")

col_in1, col_in2, col_in3 = st.columns(3)
with col_in1:
    text_input = st.text_area("📝 Text Modality (Query / Notes / Commands)", value=default_text, height=120)
with col_in2:
    vision_input = st.text_area("🖼️ Visual Modality (Image / Camera / Radiograph)", value=default_vision, height=120)
with col_in3:
    sensor_input = st.text_area("📡 Sensor Modality (Audio / LiDAR / RADAR / ECG)", value=default_sensor, height=120)

run_btn = st.button("🚀 Execute Unified Multimodal Fusion Pipeline", type="primary", use_container_width=True)

if run_btn:
    with st.spinner("Fusing multimodal representations across Early, Late, Cross-Attention, and CLIP pipelines..."):
        result = orchestrator.process_multimodal_pipeline(
            text_query=text_input,
            visual_cue=vision_input,
            sensor_input=sensor_input,
            domain_preset=domain_name
        )

        st.markdown("---")
        
        # Top Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Execution Latency", f"{result['latency_ms']} ms")
        m1.markdown('<span class="badge badge-cyan">Real-Time Fusion</span>', unsafe_allow_html=True)

        m2.metric("Early Fusion Confidence", f"{result['early_fusion']['fusion_confidence_score'] * 100:.1f}%")
        m2.markdown('<span class="badge badge-green">Input Level Concat</span>', unsafe_allow_html=True)

        m3.metric("Late Ensemble Confidence", f"{result['late_fusion']['late_ensemble_confidence'] * 100:.1f}%")
        m3.markdown(f'<span class="badge badge-pink">Dominant: {result["late_fusion"]["dominant_modality"]}</span>', unsafe_allow_html=True)

        m4.metric("Cross-Modal Grounding", f"{result['cross_attention']['cross_modal_grounding_score'] * 100:.1f}%")
        m4.markdown('<span class="badge badge-cyan">Cross-Attention Active</span>', unsafe_allow_html=True)

        # 4 Fusion Columns
        st.subheader("🔬 Multimodal Fusion Architectural Breakdown")
        f1, f2, f3, f4 = st.columns(4)

        with f1:
            st.markdown("""
            <div class="card-box">
                <h4>1. Early Fusion (Data-Level)</h4>
                <p style="font-size:0.85rem; color:#94a3b8;">Concatenates raw feature vectors into unified input.</p>
            </div>
            """, unsafe_allow_html=True)
            st.json(result['early_fusion'])

        with f2:
            st.markdown("""
            <div class="card-box">
                <h4>2. Late Fusion (Decision-Level)</h4>
                <p style="font-size:0.85rem; color:#94a3b8;">Ensembles independent unimodal model predictions.</p>
            </div>
            """, unsafe_allow_html=True)
            st.json(result['late_fusion'])

        with f3:
            st.markdown("""
            <div class="card-box">
                <h4>3. Cross-Attention Fusion</h4>
                <p style="font-size:0.85rem; color:#94a3b8;">Intermediate bidirectional token-level interaction.</p>
            </div>
            """, unsafe_allow_html=True)
            st.json(result['cross_attention'])

        with f4:
            st.markdown("""
            <div class="card-box">
                <h4>4. CLIP Contrastive Loss</h4>
                <p style="font-size:0.85rem; color:#94a3b8;">Pulls positive image-text pairs; pushes negatives.</p>
            </div>
            """, unsafe_allow_html=True)
            st.json(result['clip_contrastive'])

        # Unified LLM Solution
        st.subheader("💬 Unified Multimodal LLM Synthesis")
        st.markdown(f"""
        <div class="card-box">
            <span class="badge badge-green">Model: {result['llm_reasoning']['model']}</span>
            <span class="badge badge-cyan" style="margin-left:8px;">Reasoning Tokens: {result['llm_reasoning']['reasoning_tokens']}</span>
            <div style="margin-top:1rem; font-size:1rem; line-height:1.7;">
                {result['llm_reasoning']['solution']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Unimodal vs. Multimodal Comparison Table
        st.subheader("⚖️ Core Comparison: Unimodal vs. Multimodal Learning")
        comparison_data = [
            {"Feature": "Data Inputs", "Unimodal Learning": "Single format (Only Text or Only Vision)", "Multimodal Learning": "Multiple formats (Text + Image + Audio + Sensor)"},
            {"Feature": "Context Understanding", "Unimodal Learning": "Limited to single data perspective (72.4%)", "Multimodal Learning": "High contextual accuracy through cross-modal grounding (94.8%)"},
            {"Feature": "Complexity & Compute", "Unimodal Learning": "Lower parameter & memory overhead", "Multimodal Learning": "Higher compute (Multi-Head Cross-Attention layers)"},
            {"Feature": "Example Models", "Unimodal Learning": "Standard BERT, Original ResNet", "Multimodal Learning": "GPT-4o, Gemini, CLIP, Flamingo, Whisper"}
        ]
        st.table(comparison_data)
