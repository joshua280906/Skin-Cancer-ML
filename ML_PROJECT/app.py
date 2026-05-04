import os
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch.nn as nn

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Skin Cancer AI Detection",
    page_icon="🧬",
    layout="wide"
)

# ---------------- TITLE ----------------
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>AI Skin Cancer Detection System</h1>", unsafe_allow_html=True)
st.markdown("### Upload a dermoscopic image to analyze potential skin cancer risk.")

# ---------------- DISCLAIMER ----------------
st.warning("""
⚠️ **Medical Disclaimer**
This AI system is for research and educational purposes only.
It is NOT a substitute for professional medical diagnosis.
Always consult a certified dermatologist for clinical decisions.
""")

# ---------------- PATHS ----------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "model", "best_efficientnet_b0.pth")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at:\n{MODEL_PATH}\n\n"
            "Make sure your trained model exists inside ML_PROJECT/model/"
        )

    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)

    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)

    model.eval()
    return model

model = load_model()

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

classes = ["Benign", "Malignant"]

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Drag and Drop or Click to Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_tensor = transform(image).unsqueeze(0)  # shape: [1, 3, 224, 224]

    # Prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    predicted_class = classes[int(np.argmax(probs))]
    confidence = float(np.max(probs)) * 100

    with col2:
        st.markdown("## 🧾 Prediction Result")
        st.success(f"### {predicted_class}")
        st.info(f"Confidence: {confidence:.2f}%")

    # ---------------- PROBABILITY CHART ----------------
    st.markdown("## 📊 Prediction Probabilities")

    fig, ax = plt.subplots()
    ax.bar(classes, probs * 100)
    ax.set_ylabel("Probability (%)")
    ax.set_ylim([0, 100])
    st.pyplot(fig)

    # ---------------- RISK MESSAGE ----------------
    if predicted_class == "Malignant":
        st.error("⚠️ High risk detected. Please consult a dermatologist immediately.")
    else:
        st.success("Low risk indication. Still recommended to seek medical advice for confirmation.")