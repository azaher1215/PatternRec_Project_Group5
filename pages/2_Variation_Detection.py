from utils.layout import render_layout
import streamlit as st
from PIL import Image
from model.classifier import predict, get_model_by_name
import config

VARIATION_CLASS_MAP = {
    "Onion": ['halved', 'sliced', 'whole'],
    "Strawberry": ['Hulled', 'sliced', 'whole'],
    "Tomato": ['diced', 'vines', 'whole'],
    "Pear": ['halved', 'sliced', 'whole']
}

MODEL_PATH_MAP = {
    "Onion": config.MODEL_PATH_ONION,
    "Pear": config.MODEL_PATH_PEAR,
    "Strawberry": config.MODEL_PATH_STRAWBERRY,
    "Tomato": config.MODEL_PATH_TOMATO
}

@st.cache_resource
def load_model(product_name):
    model_path = MODEL_PATH_MAP[product_name]
    num_classes = len(VARIATION_CLASS_MAP[product_name])
    return get_model_by_name(model_path, num_classes=num_classes)

def variation_detection_page():
    st.markdown("## 🔍 Task B: Variation Detection")

    st.markdown("""
    <div class="about-box">
    This module detects variations such as <code>Whole</code>, <code>Halved</code>, <code>Diced</code>, etc. 
    for Onion, Pear, Strawberry, and Tomato using individually fine-tuned models.
    </div>
    """, unsafe_allow_html=True)

    product = st.selectbox("Select Product Type", list(MODEL_PATH_MAP.keys()))

    model = load_model(product)
    class_labels = VARIATION_CLASS_MAP[product]

    uploaded = st.file_uploader("📤 Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        label, confidence = predict(img, model, class_labels=class_labels)

        st.success(f"🔍 Detected Variation: **{label}** ({confidence * 100:.2f}% confidence)")

        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        st.image(img, caption=f"Uploaded Image - {product}", width=300)
        st.markdown("</div>", unsafe_allow_html=True)

render_layout(variation_detection_page)
