# Recipe Recommendation System - Streamlit Web App

## Overview
This is a web-based interface for our CSE 555 Pattern Recognition project focusing on recipe recommendation systems. The application is built using Streamlit and provides an intuitive interface for exploring recipes, analytics, and project documentation.

## Features

### 🏠 Home Page
- Welcome page with project overview
- Information about the system's capabilities
- Project description and objectives

### 🔍 Discover Recipes
- Recipe search and recommendation functionality (to be implemented)
- Semantic search using BERT embeddings
- Advanced filtering options
- Model file status checking

### 📊 Analytics
- Recipe analytics and statistics (to be implemented)
- User interaction patterns
- Model performance metrics

### 📋 Report
- Project documentation and report
- Links to project PDF report

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd PatternRec_Project_Group7
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run streamlit_app.py
   ```

## Deployment Options

### Option 1: Streamlit Cloud
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Select `streamlit_app.py` as your main file
5. Deploy!

### Option 2: Heroku
1. Create a `Procfile` with:
   ```
   web: sh setup.sh && streamlit run streamlit_app.py
   ```
2. Create a `setup.sh` file:
   ```bash
   mkdir -p ~/.streamlit/
   echo "[server]
   port = $PORT
   enableCORS = false
   headless = true
   " > ~/.streamlit/config.toml
   ```
3. Deploy to Heroku

### Option 3: Local Development
- Run `streamlit run streamlit_app.py`
- Access at `http://localhost:8501`

## Required Files
Make sure these model files are present in the project directory:
- `advanced_recipe_embeddings_231630.npy`
- `advanced_filtered_recipes_231630.pkl`
- `recipe_statistics_231630.pkl`
- `recipe_scores_231630.pkl`
- `tag_based_bert_model.pth`

## Project Structure
```
PatternRec_Project_Group7/
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt          # Python dependencies
├── advanced_semantic_search.py  # Core search functionality
├── NLP.py                   # NLP utilities
├── model_files/             # Trained models and embeddings
└── data/                    # Raw data files
```

## Development Notes

### Adding New Features
1. Create new functions in `streamlit_app.py`
2. Add navigation buttons if needed
3. Test locally before deployment

### Model Integration
- The app checks for model file availability
- Integration with `advanced_semantic_search.py` for search functionality
- BERT model loading and inference capabilities

## Contributing
This is a course project for CSE 555 Pattern Recognition. For questions or contributions, please refer to the project documentation.

## License
See LICENSE file for details. 