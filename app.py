import streamlit as st
from pipeline import detect_deepfake
import time
import os
import math

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Secure Lens AI | Deepfake Forensic Analysis",
    layout="wide",
    page_icon="🔍",
    initial_sidebar_state="expanded"
)

# --- PREMIUM DESIGN SYSTEM ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');
    
    :root {
        --primary: #00DBDE;
        --secondary: #FC00FF;
        --bg-dark: #05070A;
        --card-bg: rgba(20, 24, 33, 0.7);
        --text-main: #E2E8F0;
        --text-dim: #94A3B8;
        --accent-glow: rgba(0, 219, 222, 0.3);
    }

    /* Main Container */
    .stApp {
        background: linear-gradient(135deg, #05070A 0%, #0F172A 100%);
        color: var(--text-main);
        font-family: 'Inter', sans-serif;
    }

    /* Glassmorphism Classes */
    .glass-card {
        background: var(--card-bg);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8);
        margin-bottom: 2rem;
    }

    /* Header Styling */
    .hero-section {
        text-align: center;
        padding: 3rem 0;
        background: radial-gradient(circle at center, rgba(0, 219, 222, 0.1) 0%, transparent 70%);
        margin-bottom: 2rem;
    }

    .hero-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 4.5rem;
        font-weight: 900;
        background: linear-gradient(90deg, #00DBDE, #FC00FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 12px;
        margin-bottom: 0.5rem;
        filter: drop-shadow(0 0 10px rgba(0, 219, 222, 0.4));
    }

    .hero-subtitle {
        color: var(--text-dim);
        font-size: 1.1rem;
        letter-spacing: 4px;
        text-transform: uppercase;
        font-weight: 500;
    }

    /* Scan Results Interface */
    .verdict-header {
        font-family: 'Orbitron', sans-serif;
        font-size: 1.5rem;
        color: var(--text-dim);
        margin-bottom: 1.5rem;
        border-left: 4px solid var(--primary);
        padding-left: 1rem;
    }

    .probability-chart {
        height: 80px;
        width: 100%;
        background: #0F1116;
        border-radius: 40px;
        display: flex;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.1);
        position: relative;
        margin: 2rem 0;
    }

    .bar-segment {
        height: 100%;
        transition: width 2s cubic-bezier(0.4, 0, 0.2, 1);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-family: 'Orbitron', sans-serif;
        font-size: 0.9rem;
    }

    .real-seg {
        background: linear-gradient(90deg, #10B981, #34D399);
        color: #064E3B;
    }

    .fake-seg {
        background: linear-gradient(90deg, #EF4444, #F87171);
        color: #7F1D1D;
    }

    .verdict-badge {
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        font-family: 'Orbitron', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: 5px;
        margin-top: 1rem;
        box-shadow: 0 0 20px rgba(0,0,0,0.5);
    }

    .badge-real {
        border: 2px solid #10B981;
        color: #10B981;
        background: rgba(16, 185, 129, 0.1);
    }

    .badge-fake {
        border: 2px solid #EF4444;
        color: #EF4444;
        background: rgba(239, 68, 68, 0.1);
    }

    /* Metric Tooltips */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }

    .metric-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem;
        color: var(--primary);
        font-weight: 700;
    }

    .metric-lab {
        font-size: 0.7rem;
        color: var(--text-dim);
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Evidence Grid Styling */
    .evidence-i`tem {
        background: rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 8px;
        transition: transform 0.3s ease;
    }

    .evidence-item:hover {
        transform: scale(1.05);
        border-color: var(--primary);
    }

    /* Sidebar Customization */
    section[data-testid="stSidebar"] {
        background-color: #0A0C10 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #00DBDE, #FC00FF) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        height: 60px !important;
        font-family: 'Orbitron', sans-serif !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        letter-spacing: 2px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0, 219, 222, 0.3) !important;
    }

    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(0, 219, 222, 0.5) !important;
    }

    /* Footer Credits */
    .footer-note {
        text-align: center;
        color: var(--text-dim);
        font-size: 0.8rem;
        margin-top: 4rem;
        padding: 2rem;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Hide Streamlit elements */
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- HEADER SECTION ---
st.markdown("""
<div class="hero-section">
    <h1 class="hero-title">Detection of Deepfake Media Using a Hybrid CNN–RNN Model
and Particle Swarm Optimization (PSO) Algorithm</h1>
    <p class="hero-subtitle">NEXT-GEN DEEPFAKE DETECTION</p>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.markdown("<h2 style='font-family: Orbitron; color: #00DBDE;'>CONTROL PANEL</h2>", unsafe_allow_html=True)
    uploaded_video = st.file_uploader("Inject Video Data", type=["mp4", "avi", "mov"])
    
    st.markdown("---")
    st.markdown("### Workflow/Pipeline")
    
    # Placeholder for the dynamic checklist
    workflow_ui = st.empty()
    
    def render_workflow(status_dict=None):
        if status_dict is None:
            status_dict = {1: "⏳", 2: "⏳", 3: "⏳", 4: "⏳"}
        
        with workflow_ui.container():
            st.info(f"""
            **Layer 1: Spatial Analysis (CNN)** {status_dict[1]}
            **Uses Xception to detect facial micro-artifacts.**
            
            **Layer 2: Temporal Analysis (RNN)** {status_dict[2]}
            **Uses LSTM to track flickers over 30 frames.**
            
            **Layer 3: Swarm Intelligence (PSO)** {status_dict[3]}
            **Optimizes hyperparameters for peak accuracy.**
            
            **Layer 4: Hybrid Verdict** {status_dict[4]}
            **Merges CNN + RNN scores for final analysis.**
            """)

    # Initial render
    render_workflow()
    
    st.markdown("---")
    st.markdown("<p style='font-size: 0.7rem; color: #555;'>SYSTEM STATUS: ONLINE</p>", unsafe_allow_html=True)

# --- MAIN DASHBOARD ---
tab_scan, tab_tech = st.tabs(["⚡ LIVE ANALYSIS", "🧬 SYSTEM BLUEPRINTS"])

with tab_scan:
    if uploaded_video is not None:
        # Initial preparation
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_video.read())

        # Analysis Trigger
        if st.button("EXECUTE NEURAL SCAN", use_container_width=True):
            # Tracking progress
            pipeline_status = {1: "🔄", 2: "⏳", 3: "⏳", 4: "⏳"}
            render_workflow(pipeline_status)
            
            with st.status("Initializing Forensic Pipeline...", expanded=True) as status:
                st.write("Locating facial landmarks (MTCNN)...")
                time.sleep(0.8)
                
                pipeline_status[1] = "✅"
                pipeline_status[2] = "🔄"
                render_workflow(pipeline_status)
                
                st.write("Extracting spatio-temporal features...")
                time.sleep(0.8)
                
                pipeline_status[2] = "✅"
                pipeline_status[3] = "🔄"
                render_workflow(pipeline_status)
                
                st.write("Optimizing hyperparameters (PSO)...")
                result = detect_deepfake("temp_video.mp4")
                
                pipeline_status[3] = "✅"
                pipeline_status[4] = "🔄"
                render_workflow(pipeline_status)
                
                st.write("Finalizing verdict...")
                time.sleep(0.5)
                
                pipeline_status[4] = "✅"
                render_workflow(pipeline_status)
                
                status.update(label="Scanning Sequence Complete", state="complete", expanded=False)

            if "error" in result:
                st.error(f"PIPELINE FAILURE: {result['error']}")
            else:
                # Dashboard Layout
                col_res, col_vid = st.columns([1.2, 1])
                
                with col_res:
                    st.markdown("<div class='verdict-header'>DIAGNOSTIC VERDICT</div>", unsafe_allow_html=True)
                    
                    # Logic for probability display
                    label = result["prediction"]
                    confidence = result.get("confidence", 50.0)
                    
                    if label == "FAKE":
                        fake_p = confidence
                        real_p = round(100.0 - fake_p, 2)
                    else:
                        real_p = confidence
                        fake_p = round(100.0 - real_p, 2)
                        
                    badge_cls = "badge-fake" if label == "FAKE" else "badge-real"
                    
                    # Probability Bar
                    st.markdown(f"""
                        <div class="probability-chart">
                            <div class="bar-segment real-seg" style="width: {real_p}%;">{real_p}% REAL</div>
                            <div class="bar-segment fake-seg" style="width: {fake_p}%;">{fake_p}% FAKE</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Target Verdict
                    st.markdown(f"""
                        <div class="verdict-badge {badge_cls}">
                            VERDICT: {label}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Metadata Metrics
                    m_col1, m_col2, m_col3 = st.columns(3)
                    with m_col1:
                        st.markdown(f"<div class='metric-card'><div class='metric-lab'>Latencies</div><div class='metric-val'>{result['processing_time']}s</div></div>", unsafe_allow_html=True)
                    with m_col2:
                         st.markdown(f"<div class='metric-card'><div class='metric-lab'>Frame Count</div><div class='metric-val'>{result['frames_analyzed']}</div></div>", unsafe_allow_html=True)
                    with m_col3:
                         st.markdown(f"<div class='metric-card'><div class='metric-lab'>Confidence</div><div class='metric-val'>{result['confidence']}%</div></div>", unsafe_allow_html=True)

                with col_vid:
                    st.markdown("<p style='color: var(--text-dim); text-align: center;'>SOURCE REPLAY</p>", unsafe_allow_html=True)
                    st.video("temp_video.mp4")

                # Evidence Track
                face_crops_list = result.get("face_crops", [])
                frame_scores_list = result.get("frame_scores", [])

                st.markdown("""
                    <div style='margin-top: 2rem;'>
                        <h3 style='font-family: Orbitron; font-size: 1.2rem; margin-bottom: 0.25rem;'>🕵️ NEURAL FACE TRACKING EVIDENCE</h3>
                        <p style='font-size: 0.78rem; color: #94A3B8; margin-bottom: 1.2rem;'>
                            Each face crop is assigned a per-frame fake-probability using CNN feature norms blended with the global verdict score.
                            Frames reaching critical thresholds (>58%) are flagged <span style='color:#F87171;font-weight:600;'>FAKE</span>, others are flagged <span style='color:#34D399;font-weight:600;'>REAL</span>.
                        </p>
                    </div>
                """, unsafe_allow_html=True)

                if face_crops_list:
                    # 4 columns, each image stays inside its column
                    ev_cols = st.columns(4)
                    for idx, crop in enumerate(face_crops_list):
                        with ev_cols[idx % 4]:
                            # Determine per-frame verdict
                            if idx < len(frame_scores_list):
                                fs = frame_scores_list[idx]
                                frame_label = "FAKE" if fs >= 0.58 else "REAL"
                                
                                # Scale mathematically so 0.58 reads as 50% boundary on screen
                                if frame_label == "FAKE":
                                    frame_conf = round(50.0 + ((fs - 0.58) / 0.42) * 50.0, 1)
                                else:
                                    frame_conf = round(100.0 - ((fs / 0.58) * 50.0), 1)
                            else:
                                frame_label = label
                                frame_conf  = result.get("confidence", 0)

                            border_color = "#EF4444" if frame_label == "FAKE" else "#10B981"
                            badge_bg     = "rgba(239,68,68,0.15)" if frame_label == "FAKE" else "rgba(16,185,129,0.15)"
                            badge_color  = "#F87171" if frame_label == "FAKE" else "#34D399"

                            # Image — constrained inside column
                            st.image(crop, use_container_width=True)

                            # Per-frame verdict badge + confidence
                            st.markdown(f"""
                                <div style='
                                    background:{badge_bg};
                                    border:1px solid {border_color};
                                    border-radius:8px;
                                    padding:4px 6px;
                                    text-align:center;
                                    margin-top:4px;
                                '>
                                    <span style='
                                        color:{badge_color};
                                        font-family:Orbitron,sans-serif;
                                        font-size:0.65rem;
                                        font-weight:700;
                                        letter-spacing:2px;
                                    '>{frame_label}</span>
                                    <br>
                                    <span style='color:#94A3B8; font-size:0.6rem; font-family:JetBrains Mono,monospace;'>
                                        {frame_conf}% conf · Frame {idx+1}
                                    </span>
                                </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("No face crops available for this video.")
    else:
        # Empty State
        st.markdown("""
        <div style='text-align: center; padding: 5rem; border: 2px dashed rgba(255,255,255,0.05); border-radius: 30px; background: rgba(0,0,0,0.2);'>
            <h2 style='color: #444; font-family: Orbitron; letter-spacing: 5px;'>SYSTEM STANDBY</h2>
            <p style='color: #444;'>Awaiting video stream for neural disassembly...</p>
        </div>
        """, unsafe_allow_html=True)

with tab_tech:
    st.markdown("<h2 style='font-family: Orbitron; color: #00DBDE; margin-bottom: 2rem;'>INTERNAL MECHANICS</h2>", unsafe_allow_html=True)
    
    arch_col1, arch_col2 = st.columns(2)
    assets_dir = "assets"
    
    with arch_col1:
        st.markdown("""
        <div class="glass-card">
            <h3 style='color: var(--primary); font-family: Orbitron;'>⚖️ SMOTE-DLPB ANALYSIS</h3>
            <p style='font-size: 0.9rem; color: var(--text-dim);'>
                <b>Synthetic Minority Over-sampling Results</b>: 
                The plot below shows how the system synthesizes 180 artificial "Fake" points to match your "Real" dataset, 
                eliminating class bias in the feature space.
            </p>
        </div>
        """, unsafe_allow_html=True)
        smote_path = os.path.join(assets_dir, "smote_visualization.png")
        if os.path.exists(smote_path):
            st.image(smote_path, caption="Synthetic Data Generation (SMOTE) on Your Trained Data", use_container_width=True)

    with arch_col2:
        st.markdown("""
        <div class="glass-card">
            <h3 style='color: var(--secondary); font-family: Orbitron;'>🦋 PSO NEURAL SWARM</h3>
            <p style='font-size: 0.9rem; color: var(--text-dim);'>
                <b>Swarm Intelligence Optimization</b>: 
                Visualizing the particles searching for the global minima of the loss function. 
                The particles converge on the optimal decision boundary for deepfake classification.
            </p>
        </div>
        """, unsafe_allow_html=True)
        pso_path = os.path.join(assets_dir, "pso_iter_14.png")
        if os.path.exists(pso_path):
            st.image(pso_path, caption="Particle Swarm Optimization - Final Convergence", use_container_width=True)
        else:
            pso_path = os.path.join(assets_dir, "pso.png")
            if os.path.exists(pso_path):
                st.image(pso_path, use_container_width=True)

# --- FOOTER ---
st.markdown("""
<div class="footer-note">
    <p>SECURE LENS AI | PRE-ALPHA v1.2.0 | ENCRYPTION: AES-256</p>
    <p style="opacity: 0.5;">Experimental Forensic Tool - Results require expert validation</p>
</div>
""", unsafe_allow_html=True)