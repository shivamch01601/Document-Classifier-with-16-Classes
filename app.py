import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import streamlit as st

# -----------------------------------------------------------------------------
# 1. Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Document Classifier AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI Enhancement
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #F0FDF4;
        border: 2px solid #22C55E;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .prediction-title {
        color: #15803D;
        font-size: 1.2rem;
        font-weight: 600;
    }
    .prediction-value {
        color: #166534;
        font-size: 2rem;
        font-weight: 800;
        text-transform: capitalize;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Define Class Labels & Model Loader
# -----------------------------------------------------------------------------
# 16 Classes from RVL-CDIP Dataset
CLASS_NAMES = [
    'advertisement', 'budget', 'email', 'file_folder', 
    'form', 'handwritten', 'invoice', 'letter', 
    'memo', 'news_article', 'presentation', 'questionnaire', 
    'resume', 'scientific_publication', 'scientific_report', 'specification'
]

@st.cache_resource
def load_trained_model(model_path="rvl_cdip_cnn_model.pth"):
    """Loads the trained ResNet-18 model architecture and weights."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Rebuild ResNet-18 architecture matching training setup
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))
    
    # Load state dict if model file exists
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model, device, True
    else:
        # Fallback to uninitialized model if .pth isn't found yet
        model.to(device)
        model.eval()
        return model, device, False

# -----------------------------------------------------------------------------
# 3. Image Preprocessing & Inference Function
# -----------------------------------------------------------------------------
def transform_image(image):
    """Applies evaluation transformations matching ImageNet stats."""
    eval_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return eval_transforms(image).unsqueeze(0)

def predict(image, model, device):
    """Generates class probabilities for the uploaded image."""
    tensor = transform_image(image).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    return probabilities

# -----------------------------------------------------------------------------
# 4. Sidebar: Model Specifications & Settings
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/document--v1.png", width=80)
    st.title("Model Details")
    st.markdown("---")
    
    st.markdown("### 🛠 Architecture")
    st.info("**Model Base:** ResNet-18 (Transfer Learning)")
    st.markdown("**Input Size:** 224 x 224 x 3")
    st.markdown(f"**Output Classes:** {len(CLASS_NAMES)} document types")
    
    st.markdown("---")
    st.markdown("### 📊 Supported Classes")
    with st.expander("View All 16 Classes"):
        for cls in CLASS_NAMES:
            st.markdown(f"- `{cls}`")
            
    st.markdown("---")
    st.caption("Developed for RVL-CDIP Document Image Classification Pipeline.")

# -----------------------------------------------------------------------------
# 5. Main Application Interface
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">📄 Document Classifier AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload a scanned document image (invoice, form, letter, etc.) to identify its class using Deep Learning.</div>', unsafe_allow_html=True)

# Load Model
model, device, is_model_loaded = load_trained_model("rvl_cdip_cnn_model.pth")

if not is_model_loaded:
    st.warning("⚠️ Weights file `rvl_cdip_cnn_model.pth` not found in the directory! Running in demo/untrained mode.")

# Layout: Two Columns
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📤 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose an image file (JPG, PNG, TIF)", 
        type=["jpg", "jpeg", "png", "tif", "tiff"]
    )
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Document Preview", use_container_width=True)
        except Exception as e:
            st.error(f"Error opening image: {e}")

with col2:
    st.subheader("🎯 Prediction Results")
    
    if uploaded_file is not None:
        with st.spinner("Analyzing document structure..."):
            probs = predict(image, model, device)
            top_prob, top_class_idx = torch.max(probs, 0)
            predicted_class = CLASS_NAMES[top_class_idx.item()]
            confidence = top_prob.item() * 100
        
        # Display Top Predicted Class Box
        st.markdown(f"""
            <div class="prediction-card">
                <div class="prediction-title">Predicted Document Category</div>
                <div class="prediction-value">{predicted_class.replace('_', ' ').title()}</div>
                <div style="color: #4B5563; margin-top: 5px;">Confidence Score: <b>{confidence:.2f}%</b></div>
            </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        st.markdown("#### 📈 Probability Distribution (Top 5)")
        
        # Get Top 5 Predictions
        top5_prob, top5_indices = torch.topk(probs, 5)
        top5_dict = {
            CLASS_NAMES[idx.item()].replace('_', ' ').title(): prob.item() 
            for idx, prob in zip(top5_indices, top5_prob)
        }
        
        # Display progress bars for top 5
        for cls_name, p in top5_dict.items():
            st.write(f"**{cls_name}** ({p*100:.1f}%)")
            st.progress(min(float(p), 1.0))
            
    else:
        st.info("👈 Please upload an image from the left panel to trigger classification.")

# -----------------------------------------------------------------------------
# 6. Technical Overview Section
# -----------------------------------------------------------------------------
st.markdown("---")
with st.expander("ℹ️ How this app works (Technical Pipeline)"):
    st.markdown("""
    1. **Pre-processing:** The uploaded image is converted to RGB, resized to `224x224`, and normalized using ImageNet standard parameters ($mean = [0.485, 0.456, 0.406]$, $std = [0.229, 0.224, 0.225]$).
    2. **Feature Extraction:** A deep **ResNet-18 Convolutional Neural Network** processes the spatial layout, headers, lines, and text density features of the document.
    3. **Classification:** A custom Fully-Connected (FC) output layer computes raw logit outputs across all 16 RVL-CDIP classes, which are converted to probability distributions via Softmax.
    """)