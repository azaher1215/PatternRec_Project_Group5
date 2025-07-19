import streamlit as st
from utils.layout import set_custom_page_config, render_header

with open("assets/css/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

set_custom_page_config()
render_header()

st.markdown("""
<div class="about-box">
    Welcome to our Smart Kitchen Assistant — a CSE555 Final Project developed by Group 5 (Saksham & Ahmed).
    <br><br>
    🔍 This tool leverages AI to assist in:
    - Classifying images of vegetables and fruits.
    - Detecting their variations (cut, whole, sliced).
    - Recommending recipes based on natural language input.
</div>

### 🔗 Use the left sidebar to navigate between:
- 🥦 Task A: Classification
- 🧊 Task B: Variation Detection
- 🧠 NLP Recipe Recommendation
""", unsafe_allow_html=True)
