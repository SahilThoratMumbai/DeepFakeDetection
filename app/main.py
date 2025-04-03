# 1. ENVIRONMENT CONFIGURATION
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 2. CORE IMPORTS
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st

# 3. TENSORFLOW IMPORTS
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# 4. PATH CONFIGURATION
def get_project_root():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '..'))

PROJECT_ROOT = get_project_root()
sys.path.insert(0, PROJECT_ROOT)

# 5. LOCAL IMPORTS
try:
    from utils import load_models, preprocess_image, ensemble_predict
    from models.ensemble import confidence_ensemble
except ImportError as e:
    st.error(f"🔧 System Error: Module import failed - {str(e)}")
    raise

# 6. APP CONFIGURATION
st.set_page_config(
    page_title="Deepfake Forensic Analyzer",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 7. CSS LOADING (WITH UTF-8 ENCODING HANDLING)
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "styles.css")
    try:
        with open(css_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        st.error(f"⚠️ CSS Loading Error: {str(e)}")
        return ""
        
st.markdown(f"<style>{load_css()}</style>", unsafe_allow_html=True)

# SIDEBAR CONTENT (STATIC)
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-header">
                <span class="sidebar-icon">🔍</span>
                <h2>About This Tool</h2>
            </div>
            <p>Deepfake technology uses AI and deep learning to manipulate digital content.
            This detection system ensures authenticity by distinguishing real from artificially generated images.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Technology Stack
        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-header">
                <span class="sidebar-icon">🔬</span>
                <h2>Technology Stack</h2>
            </div>
            <div class="tech-stack-item">
                <span class="tech-stack-icon">🧠</span>
                <div>
                    <h3>Central Difference CNN</h3>
                    <p>Specialized in detecting local artifacts</p>
                </div>
            </div>
            <div class="tech-stack-item">
                <span class="tech-stack-icon">👁️</span>
                <div>
                    <h3>Vision Transformer</h3>
                    <p>Analyzes global image patterns</p>
                </div>
            </div>
            <div class="tech-stack-item">
                <span class="tech-stack-icon">⚖️</span>
                <div>
                    <h3>Confidence Ensemble</h3>
                    <p>Combines model outputs</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Performance Metrics
        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-header">
                <span class="sidebar-icon">📈</span>
                <h2>Performance Metrics</h2>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                <div class="metric-card">
                    <div class="value">98.28%</div>
                    <div class="label">Accuracy</div>
                </div>
                <div class="metric-card">
                    <div class="value">98.56%</div>
                    <div class="label">Precision</div>
                </div>
                <div class="metric-card">
                    <div class="value">97.87%</div>
                    <div class="label">Recall</div>
                </div>
                <div class="metric-card">
                    <div class="value">98.65%</div>
                    <div class="label">Specificity</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Disclaimer
        st.markdown("""
        <div class="notice-box">
            <h4>Important Notice</h4>
            <p>Results should be used as part of a comprehensive analysis process with human review.
</p>
        </div>
        """, unsafe_allow_html=True)

# MAIN CONTENT
def main():
    # Render static sidebar immediately
    render_sidebar()

    # Header Section
    st.markdown("""
    <div class="app-header">
        <div class="header-content">
            <h1>🕵️ Deepfake Forensic Analyzer</h1>
            <p class="tagline">Advanced AI Detection Suite</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state for models
    if 'models_loaded' not in st.session_state:
        with st.spinner("🔍 Initializing forensic analysis engines..."):
            try:
                st.session_state.cnn_model, st.session_state.vit_model = load_models()
                st.session_state.models_loaded = True
            except Exception as e:
                st.error(f"🚨 System Error: {str(e)}")
                return

    # File Upload
    uploaded_file = st.file_uploader(
        "📁 Drag & drop or select image file",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG (Max 10MB)"
    )

    if not uploaded_file:
        return

    # Create layout columns
    col1, col2 = st.columns([1, 1.2], gap="large")

    try:
        # Image Processing (Left Column)
        with col1:
            st.subheader("📸 Submitted Evidence")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True, caption="Original Image")

        # Prediction (Right Column)
        with col2:
            st.subheader("🔍 Forensic Analysis Report")
            st.caption("Digital Authenticity Assessment")
            
            with st.spinner("🧠 Analyzing digital fingerprints..."):
                image_array = preprocess_image(image)
                result = ensemble_predict(st.session_state.cnn_model, st.session_state.vit_model, image_array)
                
                # Verdict Card
                is_real = result['final_label'] == "Real"
                verdict_class = "real" if is_real else "fake"
                verdict_icon = "✅" if is_real else "❌"
                verdict_text = "Real Image" if is_real else "Deepfake Image"
                
                st.markdown(
                    f"""
                    <div class="verdict-card {verdict_class}">
                        <div class="verdict-content">
                            <div class="verdict-title">FINAL VERDICT</div>
                            <div class="verdict-result">
                                <span>{verdict_icon}</span>
                                <span>{verdict_text}</span>
                            </div>
                            <div class="confidence-display">
                                <span class="confidence-value">{result['final_confidence']:.1%}</span>
                                <span class="confidence-label">confidence score</span>
                            </div>
                        </div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

                # Model Analysis
                st.subheader("🧬 Model Analysis Breakdown")
                
                for model in ['cnn_pred', 'vit_pred']:
                    icon = "🧠" if model == 'cnn_pred' else "👁️"
                    name = "Convolutional Network" if model == 'cnn_pred' else "Vision Transformer"
                    
                    st.markdown(
                        f"""
                        <div class="model-analysis">
                            <div class="model-header">
                                <span class="model-icon">{icon}</span>
                                <h3>{name}</h3>
                            </div>
                            <div class="analysis-result">
                                <span class="result-label">Finding:</span>
                                <span class="result-value {result[model]['label'].lower()}">{result[model]['label']}</span>
                            </div>
                            <div class="confidence-meter">
                                <div class="meter-bar">
                                    <div class="meter-fill" style="width:{result[model]['confidence']*100}%"></div>
                                </div>
                                <div class="confidence-value">{result[model]['confidence']:.1%} confidence</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Confidence Visualization
                st.subheader("📊 Confidence Metrics")
                fig, ax = plt.subplots(figsize=(10, 4))
                models = ['CNN', 'ViT', 'Final Verdict']
                confidences = [
                    result['cnn_pred']['confidence'],
                    result['vit_pred']['confidence'],
                    result['final_confidence']
                ]
                colors = ['#4285F4', '#EA4335', '#34A853'] if is_real else ['#EA4335', '#4285F4', '#FBBC05']
                
                bars = ax.bar(models, confidences, color=colors, width=0.6)
                ax.set_ylim(0, 1.15)
                ax.set_ylabel('Confidence Level')
                ax.set_title('Model Confidence Comparison')
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                            f'{height:.1%}',
                            ha='center', va='bottom')
                st.pyplot(fig)

    except Exception as e:
        st.error(f"🔧 Analysis Error: {str(e)}")

if __name__ == "__main__":
    main()