import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Recipe Recommendation System",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5em;
        margin-bottom: 30px;
        padding: 20px;
        background: linear-gradient(90deg, #f0f2f6, #ffffff);
        border-radius: 10px;
    }
    
    .nav-link {
        padding: 10px 20px;
        margin: 0 10px;
        background-color: #f0f2f6;
        border-radius: 5px;
        text-decoration: none;
        color: #1f77b4;
        font-weight: bold;
    }
    
    .nav-link:hover {
        background-color: #1f77b4;
        color: white;
    }
    
    .section-header {
        color: #1f77b4;
        font-size: 2em;
        margin-top: 30px;
        margin-bottom: 20px;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 10px;
    }
    
    .empty-section {
        background-color: #f8f9fa;
        padding: 40px;
        border-radius: 10px;
        text-align: center;
        color: #6c757d;
        font-style: italic;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Main header
    st.markdown('<h1 class="main-header">🍽️ Recipe Recommendation System</h1>', unsafe_allow_html=True)
    
    # Navigation bar
    st.markdown("### Navigation")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.page = "home"
    
    with col2:
        if st.button("🔍 Discover Recipes", use_container_width=True):
            st.session_state.page = "discover"
    
    with col3:
        if st.button("📊 Analytics", use_container_width=True):
            st.session_state.page = "analytics"
    
    with col4:
        if st.button("📋 Report", use_container_width=True):
            st.session_state.page = "report"
    
    # Initialize session state for page navigation
    if 'page' not in st.session_state:
        st.session_state.page = "home"
    
    st.markdown("---")
    
    # Page routing
    if st.session_state.page == "home":
        show_home_page()
    elif st.session_state.page == "discover":
        show_discover_page()
    elif st.session_state.page == "analytics":
        show_analytics_page()
    elif st.session_state.page == "report":
        show_report_page()

def show_home_page():
    """Home page - currently empty as requested"""
    st.markdown('<h2 class="section-header">🏠 Welcome to Recipe Recommendation System</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="empty-section">
        <h3>Welcome!</h3>
        <p>This is the home page of our Recipe Recommendation System.</p>
        <p>Content will be added here soon...</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add some basic project information
    st.markdown("### About This Project")
    st.info("""
    This is a CSE 555 Pattern Recognition project focusing on recipe recommendation systems.
    The system uses advanced machine learning techniques including:
    - BERT embeddings for semantic understanding
    - Advanced filtering and scoring algorithms
    - Neural network models for pattern recognition
    """)

def show_discover_page():
    """Discover recipes page - will contain recipe search functionality"""
    st.markdown('<h2 class="section-header">🔍 Discover Recipes</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="empty-section">
        <h3>Recipe Discovery</h3>
        <p>This section will contain recipe search and recommendation functionality.</p>
        <p>Features to be implemented:</p>
        <ul style="text-align: left; display: inline-block;">
            <li>Semantic recipe search</li>
            <li>Ingredient-based filtering</li>
            <li>Recipe recommendations</li>
            <li>Advanced filtering options</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Show file status
    st.markdown("### Available Model Files")
    model_files = [
        "advanced_recipe_embeddings_231630.npy",
        "advanced_filtered_recipes_231630.pkl",
        "recipe_statistics_231630.pkl",
        "recipe_scores_231630.pkl",
        "tag_based_bert_model.pth"
    ]
    
    for file in model_files:
        if Path(file).exists():
            st.success(f"✅ {file} - Ready")
        else:
            st.error(f"❌ {file} - Not found")

def show_analytics_page():
    """Analytics page - empty section as requested"""
    st.markdown('<h2 class="section-header">📊 Analytics</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="empty-section">
        <h3>Recipe Analytics</h3>
        <p>This section will contain analytics and statistics about recipes and user interactions.</p>
        <p>Planned features:</p>
        <ul style="text-align: left; display: inline-block;">
            <li>Recipe popularity trends</li>
            <li>Ingredient usage statistics</li>
            <li>User interaction patterns</li>
            <li>Model performance metrics</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def show_report_page():
    """Report page - empty as requested"""
    st.markdown('<h2 class="section-header">📋 Report</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="empty-section">
        <h3>Project Report</h3>
        <p>This section will contain the project report and documentation.</p>
        <p>Report sections to be added:</p>
        <ul style="text-align: left; display: inline-block;">
            <li>Project overview and objectives</li>
            <li>Methodology and approach</li>
            <li>Results and evaluation</li>
            <li>Conclusions and future work</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Show PDF report if available
    if Path("CSE_455_555_Term_Project_VF.pdf").exists():
        st.markdown("### Project PDF Report")
        st.info("📄 Project report PDF is available in the repository")

if __name__ == "__main__":
    main() 