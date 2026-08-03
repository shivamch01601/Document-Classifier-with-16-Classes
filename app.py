import os
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import onnxruntime as ort

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM STYLING
# ==========================================
st.set_page_config(
    page_title="RVL-CDIP Document Classifier",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast CSS (Fixes dark mode text-on-text readability)
st.markdown("""
    <style>
    /* Global Page Structure */
    .main {
        padding: 1.5rem;
    }
    
    /* Custom Result Card */
    .prediction-card {
        background-color: #0f172a;
        color: #ffffff;
        padding: 1.25rem;
        border-radius: 8px;
        border-left: 6px solid #2563eb;
        margin-bottom: 1rem;
    }
    .prediction-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 0.25rem;
    }
    .prediction-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.25rem;
    }
    .prediction-confidence {
        font-size: 0.95rem;
        color: #38bdf8;
        font-weight: 600;
    }
    
    /* Review Box */
    .review-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
        color: #1e293b;
    }
    
    @media (prefers-color-scheme: dark) {
        .review-box {
            background-color: #1e293b;
            border-color: #334155;
            color: #f8fafc;
        }
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CONSTANTS & CLASS MAP
# ==========================================
ONNX_MODEL_PATH = "rvl_cdip_efficientnet_best.onnx"

CLASS_NAMES = [
    "advertisement", "budget", "email", "file_folder",
    "form", "handwritten", "invoice", "letter",
    "memo", "news_article", "presentation", "questionnaire",
    "resume", "scientific_publication", "scientific_report", "specification"
]

CLASS_DESCRIPTIONS = {
    "Advertisement": "Promotional material, flyers, or magazine print ads.",
    "Budget": "Financial planning sheets, ledger sheets, and cost spreadsheets.",
    "Email": "Printed digital email correspondence headers and body threads.",
    "File Folder": "Cover jackets, index labels, or binder divider pages.",
    "Form": "Structured template documents with blank entry fields or checkboxes.",
    "Handwritten": "Cursive or printed pen/pencil notes and personal correspondence.",
    "Invoice": "Commercial bills of sale, itemized line items, and payment requests.",
    "Letter": "Formal typed business correspondence with letterheads.",
    "Memo": "Internal corporate communications and memorandum headers.",
    "News Article": "Newspaper clippings, press releases, or journalistic columns.",
    "Presentation": "Slide decks, overhead transparency prints, or landscape diagrams.",
    "Questionnaire": "Survey sheets, multiple-choice forms, and feedback questionnaires.",
    "Resume": "Curriculum vitae, candidate job histories, and background summaries.",
    "Scientific Publication": "Peer-reviewed journal papers with formal multi-column layouts.",
    "Scientific Report": "Technical laboratory summaries, technical figures, and data logs.",
    "Specification": "Engineering blueprints, product technical sheets, or requirement lists."
}

IMG_SIZE = (260, 260)  # Native input resolution for EfficientNet-B2

# ==========================================
# 3. MODEL LOADING (CACHED)
# ==========================================
@st.cache_resource
def load_onnx_model(model_path: str):
    """Loads and caches the ONNX Runtime inference session."""
    if not os.path.exists(model_path):
        return None
    providers = ['CPUExecutionProvider']
    session = ort.InferenceSession(model_path, providers=providers)
    return session

# ==========================================
# 4. PREPROCESSING & UTILS
# ==========================================
def preprocess_image(image: Image.Image) -> np.ndarray:
    """Standardizes input images to match EfficientNet-B2 validation pipeline."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    # ImageNet Mean & Standard Deviation
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    img_array = (img_array - mean) / std
    img_array = img_array.transpose(2, 0, 1)
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    
    return img_array

def softmax(logits: np.ndarray) -> np.ndarray:
    """Computes softmax probabilities from raw logit scores."""
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

# ==========================================
# 5. MAIN APPLICATION INTERFACE
# ==========================================
def main():
    st.title("RVL-CDIP Document Image Classifier")
    st.caption("Automated Deep Learning Visual Analysis System | EfficientNet-B2 + ONNX Engine")
    st.markdown("---")
    
    # Sidebar Setup
    st.sidebar.header("System Metadata")
    st.sidebar.markdown("""
    * **Architecture:** EfficientNet-B2
    * **Input Resolution:** 260 × 260 px
    * **Inference Runtime:** ONNX Execution Engine
    * **Target Classes:** 16 Document Categories
    """)
    st.sidebar.markdown("---")
    
    # Model Architecture Info Box in Sidebar
    with st.sidebar.expander("About the Model"):
        st.markdown("""
        **Training Configuration:**
        - **Backbone:** Pre-trained EfficientNet-B2 with Squeeze-and-Excitation (SE) attention blocks.
        - **Dataset:** RVL-CDIP (Ryerson Vision Lab Document Dataset).
        - **Optimization:** AdamW optimizer with Cosine Annealing learning rate schedule.
        - **Loss:** Cross-Entropy with Label Smoothing (0.1).
        """)

    session = load_onnx_model(ONNX_MODEL_PATH)
    
    if session is None:
        st.error(f"Asset Error: Model file '{ONNX_MODEL_PATH}' not found in working directory. Run the training script to export the ONNX model.")
        st.stop()
        
    st.sidebar.success("Model Status: Operational")
    
    # Main File Uploader
    uploaded_file = st.file_uploader(
        "Upload a document image for category classification", 
        type=["png", "jpg", "jpeg", "tiff", "bmp", "webp"]
    )
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            
            # Layout Columns
            col_img, col_metrics = st.columns([1, 1], gap="large")
            
            with col_img:
                st.subheader("Source Document")
                st.image(image, use_container_width=True)
                
            with col_metrics:
                st.subheader("Model Prediction")
                
                with st.spinner("Executing visual layout analysis..."):
                    input_tensor = preprocess_image(image)
                    
                    input_name = session.get_inputs()[0].name
                    output_name = session.get_outputs()[0].name
                    logits = session.run([output_name], {input_name: input_tensor})[0]
                    
                    probabilities = softmax(logits)[0]
                    top_idx = int(np.argmax(probabilities))
                    top_class = CLASS_NAMES[top_idx].replace("_", " ").title()
                    top_confidence = probabilities[top_idx] * 100

                # High-Contrast HTML Card (Avoids poor dark mode text rendering)
                st.markdown(f"""
                <div class="prediction-card">
                    <div class="prediction-title">Primary Category Classification</div>
                    <div class="prediction-value">{top_class}</div>
                    <div class="prediction-confidence">{top_confidence:.2f}% Confidence Score</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.progress(float(probabilities[top_idx]))
                
                st.markdown("### Category Probability Breakdown")
                
                # Ranked Probability Distribution
                top_3_indices = np.argsort(probabilities)[::-1][:3]
                for rank, idx in enumerate(top_3_indices, start=1):
                    c_name = CLASS_NAMES[idx].replace("_", " ").title()
                    score = probabilities[idx] * 100
                    st.write(f"**{rank}. {c_name}** — `{score:.2f}%`")
                
                st.markdown("---")
                
                # ==========================================
                # 6. USER REVIEW & VERIFICATION SYSTEM
                # ==========================================
                st.subheader("User Review & Verification")
                st.caption("Review the prediction to log feedback or correct misclassifications.")
                
                review_choice = st.radio(
                    "Is the model prediction correct?",
                    ["Select an option", "Yes, prediction is accurate", "No, incorrect category"],
                    index=0,
                    key="review_radio"
                )
                
                if review_choice == "Yes, prediction is accurate":
                    st.success("Verification logged: Prediction confirmed as correct.")
                    
                elif review_choice == "No, incorrect category":
                    correct_class = st.selectbox(
                        "Please select the correct document category:",
                        [c.replace("_", " ").title() for c in CLASS_NAMES],
                        index=top_idx
                    )
                    if st.button("Submit Feedback"):
                        st.info(f"Feedback logged: Image flagged as '{correct_class}' (Predicted: '{top_class}').")

        except Exception as e:
            st.error(f"Image Processing Error: {e}")
            
    # Reference Guide Drawer at Page Bottom
    st.markdown("---")
    with st.expander("Supported Document Categories Reference Guide"):
        st.write("Supported class definitions in the RVL-CDIP taxonomy:")
        guide_cols = st.columns(2)
        items = list(CLASS_DESCRIPTIONS.items())
        
        with guide_cols[0]:
            for key, desc in items[:8]:
                st.markdown(f"**{key}:** {desc}")
                
        with guide_cols[1]:
            for key, desc in items[8:]:
                st.markdown(f"**{key}:** {desc}")

if __name__ == "__main__":
    main()