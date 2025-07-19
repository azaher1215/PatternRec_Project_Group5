# UI Pattern Project

## Overview
This project is a multi-featured application focused on food image classification, variation detection, recipe recommendation, and reporting. It leverages deep learning and NLP techniques to provide a comprehensive toolkit for food-related data analysis and user interaction.

## Features
- **Image Classification:** Classify food images using pre-trained models.
- **Variation Detection:** Detect variations in food items.
- **Recipe Recommendation:** Recommend recipes based on user input and image analysis.
- **Report Generation:** Generate reports based on classification and recommendation results.

## Project Structure
```
UI_pattern/
├── assets/
│   ├── css/                # Stylesheets
│   ├── modelWeights/       # Pre-trained model weights (.pth)
│   └── nlp/                # NLP data and models
├── config.py               # Configuration file
├── Home.py                 # Main entry point (possibly Streamlit or similar)
├── model/                  # Model code (classifier, recipe search)
├── pages/                  # App pages (image classification, variation detection, etc.)
├── sakenv/                 # Python virtual environment
├── utils/                  # Utility functions (layout, etc.)
```

## Setup Instructions
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd UI_pattern
   ```
2. **Create and activate the virtual environment:**
   (Already included as `sakenv/`)
   ```bash
   source sakenv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the application:**
   - If using Streamlit:
     ```bash
     streamlit run Home.py
     ```
   - Or follow the instructions in `Home.py`.

## Python Version
- Python 3.12.2

## Notes
- Model weights and NLP data are stored in the `assets/` directory.
- Ensure you have the necessary permissions to access large files in `assets/modelWeights/` and `assets/nlp/`.
- For best results, use the provided virtual environment and requirements file. 
