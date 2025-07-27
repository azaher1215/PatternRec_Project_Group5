# layout.py
import streamlit as st

def set_custom_page_config():
    st.set_page_config(
        page_title="Computer Vision and Natural Language Processing Project",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def render_header():
    st.markdown("""
        <div class="project-header">
            <h1>Computer Vision and Natural Language Processing Project</h1>
            <p>CSE555 Final Project — Group 5: Saksham Lakhera & Ahmed Zaher</p>
        </div>
    """, unsafe_allow_html=True)

def render_footer():
    st.markdown("""
        <div class="footer">
            <p>Made with ❤️ by Saksham & Zaher | CSE555 @ UB</p>
        </div>
    """, unsafe_allow_html=True)

def render_layout(content_function):
    set_custom_page_config()
    with open("assets/css/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    render_header()
    content_function()
    render_footer()
