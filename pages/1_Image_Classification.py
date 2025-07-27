from utils.layout import render_layout
import streamlit as st
from PIL import Image
from model.classifier import get_model, predict

def classification_page():
    st.markdown("## 🖼️ Task A: Image Classification")

    st.markdown("""
    <div class="about-box">
    This module classifies images into <b>Onion, Pear, Strawberry, or Tomato</b>
    using an EfficientNet-B0 model.
    </div>
    """, unsafe_allow_html=True)

    model = load_model()

    uploaded = st.file_uploader("📤 Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        label, confidence = predict(img, model)
        print(label)

        st.success(f"🎯 Prediction: **{label.upper()}** ({confidence*100:.2f}% confidence)")

        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        st.image(img, caption="Uploaded Image", width=300)
        st.markdown("</div>", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return get_model()

render_layout(classification_page)
