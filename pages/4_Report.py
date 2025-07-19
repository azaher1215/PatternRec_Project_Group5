import streamlit as st

def render_report():
    st.title("📊 Recipe Search System Report")

    st.markdown("""
        ## Overview
        This report summarizes the working of the **custom BERT-based Recipe Recommendation System**, dataset characteristics, scoring algorithm, and evaluation metrics.
    """)

    st.markdown("### 🔍 Query Embedding and Similarity Calculation")
    st.latex(r"""
        \text{Similarity}(q, r_i) = \cos(\hat{q}, \hat{r}_i) = \frac{\hat{q} \cdot \hat{r}_i}{\|\hat{q}\|\|\hat{r}_i\|}
    """)
    st.markdown("""
        Here, $\\hat{q}$ is the BERT embedding of the query, and $\\hat{r}_i$ is the embedding of the i-th recipe.
    """)

    st.markdown("### 🏆 Final Score Calculation")
    st.latex(r"""
        \text{Score}_i = 0.6 \times \text{Similarity}_i + 0.4 \times \text{Popularity}_i
    """)

    st.markdown("### 📊 Dataset Summary")
    st.markdown("""
        - **Total Recipes:** 231,630  
        - **Average Tags per Recipe:** ~6  
        - **Ingredients per Recipe:** 3 to 20  
        - **Ratings Data:** Extracted from user interaction dataset  
    """)

    st.markdown("### 🧪 Evaluation Strategy")
    st.markdown("""
        We use a combination of:
        - Manual inspection
        - Recipe diversity analysis
        - Match vs rating correlation
        - Qualitative feedback from test queries
    """)

    st.markdown("---")
    st.markdown("© 2025 Your Name. All rights reserved.")

# If using a layout wrapper:
render_report()



# LaTeX content as string
latex_report = r"""
\documentclass{article}
\usepackage{amsmath}
\usepackage{geometry}
\geometry{margin=1in}
\title{Recipe Recommendation System Report}
\author{Saksham Lakhera}
\date{\today}

\begin{document}
\maketitle

\section*{Overview}
This report summarizes the working of the \textbf{custom BERT-based Recipe Recommendation System}, dataset characteristics, scoring algorithm, and evaluation metrics.

\section*{Query Embedding and Similarity Calculation}
\[
\text{Similarity}(q, r_i) = \cos(\hat{q}, \hat{r}_i) = \frac{\hat{q} \cdot \hat{r}_i}{\|\hat{q}\|\|\hat{r}_i\|}
\]
Here, $\hat{q}$ is the BERT embedding of the query, and $\hat{r}_i$ is the embedding of the i-th recipe.

\section*{Final Score Calculation}
\[
\text{Score}_i = 0.6 \times \text{Similarity}_i + 0.4 \times \text{Popularity}_i
\]

\section*{Dataset Summary}
\begin{itemize}
  \item \textbf{Total Recipes:} 231,630
  \item \textbf{Average Tags per Recipe:} $\sim$6
  \item \textbf{Ingredients per Recipe:} 3 to 20
  \item \textbf{Ratings Source:} User interaction dataset
\end{itemize}

\section*{Evaluation Strategy}
We use a combination of:
\begin{itemize}
  \item Manual inspection
  \item Recipe diversity analysis
  \item Match vs rating correlation
  \item Qualitative user feedback
\end{itemize}

\end{document}
"""

# ⬇️ Download button to get the .tex file
st.markdown("### 📥 Download LaTeX Report")
st.download_button(
    label="Download LaTeX (.tex)",
    data=latex_report,
    file_name="recipe_report.tex",
    mime="text/plain"
)

# 📤 Optional: Show the .tex content in the app
with st.expander("📄 View LaTeX (.tex) File Content"):
    st.code(latex_report, language="latex")
