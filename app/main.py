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

# 7. CSS LOADING
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "styles.css")
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            return f.read()
    return ""
st.markdown(f"<style>{load_css()}</style>", unsafe_allow_html=True)

# MAIN APPLICATION
def main():
    # Header Section
    st.markdown("""
    <div class="app-header">
        <div class="header-content">
            <h1>🕵️ Deepfake Forensic Analyzer</h1>
            <p class="tagline">Advanced AI Detection Suite</p>
        </div>
        <div class="header-description">
            <p>Utilizing cutting-edge ensemble modeling to identify synthetic media</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Model Loading
    with st.spinner("🔍 Initializing forensic analysis engines..."):
        try:
            cnn_model, vit_model = load_models()
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

    col1, col2 = st.columns([1, 1.2], gap="large")

    try:
        # Image Processing
        with col1:
            st.subheader("📸 Submitted Evidence")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True, caption="Original Image")

        # Prediction
        with st.spinner("🧠 Analyzing digital fingerprints..."):
            image_array = preprocess_image(image)
            result = ensemble_predict(cnn_model, vit_model, image_array)

        # Results Display
        with col2:
            # Forensic Report Header
            st.subheader("🔍 Forensic Analysis Report")
            st.caption("Digital Authenticity Assessment")
            
            # Verdict Card
            verdict_class = "real" if result['final_label'] == "Real" else "fake"
            verdict_icon = "✅ Authentic" if result['final_label'] == "Real" else "❌ Synthetic"
            st.markdown(
                f"""
                <div class="verdict-card {verdict_class}">
                    <div class="verdict-content">
                        <div class="verdict-title">FINAL VERDICT</div>
                        <div class="verdict-result">{verdict_icon}</div>
                        <div class="confidence-display">
                            <span class="confidence-value">{result['final_confidence']:.1%}</span>
                            <span class="confidence-label">confidence score</span>
                        </div>
                    </div>
                </div>
                """, 
                unsafe_allow_html=True
            )

            # Model Analysis Section
            st.subheader("🧬 Model Analysis Breakdown")
            
            # CNN Analysis
            st.markdown(
                f"""
                <div class="model-analysis">
                    <div class="model-header">
                        <span class="model-icon">🧠</span>
                        <h3>Convolutional Network</h3>
                    </div>
                    <div class="analysis-result">
                        <span class="result-label">Finding:</span>
                        <span class="result-value {result['cnn_pred']['label'].lower()}">{result['cnn_pred']['label']}</span>
                    </div>
                    <div class="confidence-meter">
                        <div class="meter-bar">
                            <div class="meter-fill" style="width:{result['cnn_pred']['confidence']*100}%"></div>
                        </div>
                        <div class="confidence-value">{result['cnn_pred']['confidence']:.1%} confidence</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # ViT Analysis
            st.markdown(
                f"""
                <div class="model-analysis">
                    <div class="model-header">
                        <span class="model-icon">👁️</span>
                        <h3>Vision Transformer</h3>
                    </div>
                    <div class="analysis-result">
                        <span class="result-label">Finding:</span>
                        <span class="result-value {result['vit_pred']['label'].lower()}">{result['vit_pred']['label']}</span>
                    </div>
                    <div class="confidence-meter">
                        <div class="meter-bar">
                            <div class="meter-fill" style="width:{result['vit_pred']['confidence']*100}%"></div>
                        </div>
                        <div class="confidence-value">{result['vit_pred']['confidence']:.1%} confidence</div>
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
            colors = ['#4285F4', '#EA4335', '#34A853']
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

    # Sidebar - Now using native Streamlit components
    with st.sidebar:
        st.header("🔍 About This Tool")
        
        # Technology Stack
        st.subheader("🔬 Technology Stack")
        st.markdown("""
        - **Central Difference CNN:** Specialized in artifact detection
        - **Vision Transformer:** Analyzes global image patterns
        - **Confidence Ensemble:** Combines models for higher accuracy
        """)
        
        # Performance Metrics
        st.subheader("📈 Performance Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", "94.2%")
            st.metric("Avg. Runtime", "89ms")
        with col2:
            st.metric("AUC Score", "0.96")
            st.metric("Precision", "98%")
        
        # Disclaimer
        st.warning("""
        **⚠️ Important Notice**  
        Results should be used as part of a comprehensive analysis process with human review.
        """)

if __name__ == "__main__":
    main()