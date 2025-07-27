# CSE 555 - Introduction to Pattern Recognition
### by: Ahmed Zaher and Saksham Lakhera

## Overview
This project is a multi-featured application focused on food image classification, variation detection, recipe recommendation, and a report that combines the analysis and discusses the resutls. It leverages deep learning and NLP techniques to provide a comprehensive toolkit for food-related data analysis and user interaction.

## Features
- **Image Classification:** Classify food images using pre-trained models.
- **Variation Detection:** Detect variations in food items.
- **Recipe Recommendation:** Recommend recipes based on user input and image analysis.
- **Report section:** View or download the scietific report that contains all the analysis .

## Interactive Websitelink
To run the website, click on [this](https://huggingface.co/spaces/PatternGroup5/pattern). 
The website is hosted on Hugging face due to size. 

## Datasets
- The pictures used to train the CV part can be found on [dropbox](https://www.dropbox.com/scl/fo/19579q8wzr4crnox1a7dj/AIRrxlllLPTkA7lt08MMno0?rlkey=xyqtykyc6css2k644gk7ewv5f&st=i1fjkcq5&dl=0)
- The dataset used for training the NLP section along with the final weights and a complete recipe embeddings can be found on google [drive](https://drive.google.com/drive/folders/1m6cfy4NuxIKNDBtJqm150NNN0FSUS8Np?usp=sharing).
- Model weights data for CV and all other assets are stored in the `assets/` directory.

## Project Structure
```
PatternRec_Project_Group5/
├── assets/
│   ├── css/                # Stylesheets
│   ├── modelWeights/       # Pre-trained model weights (.pth)
│   ├── images/             # report assets
│   ├── pdf/                # report pdf
│   └── nlp/                # NLP data and models
├── config.py               # Configuration file
├── Home.py                 # Main entry point (possibly Streamlit or similar)
├── model/                  # Model code (classifier, recipe search)
├── pages/                  # App pages (image classification, variation detection, etc.)
├── utils/                  # Utility functions (layout, etc.)
├── requirements.txt/       # Package requirements to run this repo
├── README_huggingface.md/  # Necessary to run the project on Hugging Face
├── scripts/                # Scripts used to train and test the dataset 


```

## Setup Instructions
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd PatternRec_Project_Group5
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the application:**
   Using Streamlit:
   ```bash
   streamlit run Home.py
   ```
