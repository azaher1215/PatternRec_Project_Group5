#!/usr/bin/env python3
"""
Streamlit Recipe Recommendation App - Google Drive Version
=========================================================

Clean, fast recipe search using PyTorch embeddings and trained BERT model.
Loads large files from Google Drive for deployment on Streamlit Cloud.
Completely avoids numpy/pandas dependency issues.
"""

import streamlit as st
import torch
from transformers import BertTokenizer, BertModel
import pickle
import os
import csv
from typing import List, Dict
import time
import ast
import requests
import gdown
from pathlib import Path
import tempfile

# Google Drive file IDs - UPDATE THESE WITH YOUR ACTUAL FILE IDs
GOOGLE_DRIVE_FILES = {
    'torch_recipe_embeddings_231630.pt': '1PSidY1toSfgECXDxa4pGza56Jq6vOq6t',  # Update with your actual ID
    'tag_based_bert_model.pth': '1LBl7yFs5JFqOsgfn88BF9g83W9mxiBm6',
    'RAW_recipes.csv': '1rFJQzg_ErwEpN6WmhQ4jRyiXv6JCINyf',
    'recipe_statistics_231630.pkl': '6EA_usv59CCCU1IXqtuM7i084E',
    'recipe_scores_231630.pkl': '1gfPBzghKHOZqgJu4VE9NkandAd6FGjrA'
}

# Page config
st.set_page_config(
    page_title="Group 5 Pattern Recognition Project",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1976d2, #42a5f5);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .recipe-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .recipe-title {
        color: #1976d2;
        font-size: 1.3em;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .recipe-meta {
        color: #6c757d;
        font-size: 0.9em;
        margin: 0.5rem 0;
    }
    .score-badge {
        background: #28a745;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 10px;
        font-size: 0.8em;
        margin-right: 0.5rem;
    }
    .score-overall {
        background: #ffc107;
    }
    .recipe-tag {
        background: #e3f2fd;
        color: #1565c0;
        padding: 0.2rem 0.4rem;
        border-radius: 8px;
        font-size: 0.8em;
        margin: 0.1rem;
        display: inline-block;
    }
    .download-status {
        background: #e3f2fd;
        color: #1565c0;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #bbdefb;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def download_file_from_drive(file_id: str, destination: str, file_name: str) -> bool:
    """Download file from Google Drive with progress tracking"""
    try:
        # Create a progress placeholder
        progress_placeholder = st.empty()
        
        with progress_placeholder.container():
            st.info(f"📥 Downloading {file_name} from Google Drive...")
            progress_bar = st.progress(0)
            
            # Download with gdown
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, destination, quiet=False)
            
            progress_bar.progress(100)
            st.success(f"✅ Downloaded {file_name}")
        
        # Clear progress indicators
        progress_placeholder.empty()
        return True
        
    except Exception as e:
        st.error(f"❌ Error downloading {file_name}: {str(e)}")
        st.error("Please check that the Google Drive file is publicly accessible")
        return False

def ensure_files_downloaded():
    """Ensure all required files are downloaded from Google Drive"""
    st.info("🔍 Checking required files...")
    
    download_needed = []
    
    # Check which files need to be downloaded
    for filename, file_id in GOOGLE_DRIVE_FILES.items():
        if not os.path.exists(filename):
            if file_id == 'YOUR_MODEL_FILE_ID_HERE' or file_id == 'YOUR_RECIPES_FILE_ID_HERE':
                st.warning(f"⚠️ {filename} not configured - please update Google Drive file ID")
                continue
            download_needed.append((filename, file_id))
    
    if not download_needed:
        st.success("✅ All files ready!")
        return True
    
    # Download missing files
    st.info(f"📥 Downloading {len(download_needed)} files from Google Drive...")
    success_count = 0
    
    for filename, file_id in download_needed:
        if download_file_from_drive(file_id, filename, filename):
            success_count += 1
    
    if success_count == len(download_needed):
        st.success("✅ All files downloaded successfully!")
        return True
    else:
        st.error(f"❌ Failed to download {len(download_needed) - success_count} files")
        return False

class GoogleDriveRecipeSearch:
    """
    Recipe search with Google Drive file loading and numpy-free implementation
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Download files from Google Drive
        if not ensure_files_downloaded():
            st.error("❌ Failed to download required files from Google Drive")
            self.is_ready = False
            return
        
        # Initialize tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        
        # Load trained model if available
        if os.path.exists('tag_based_bert_model.pth'):
            self.model.load_state_dict(torch.load('tag_based_bert_model.pth', map_location=self.device))
            st.success("🧠 Trained BERT model loaded successfully!")
        else:
            st.warning("⚠️ Using pre-trained BERT (trained model not found)")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Load data without pandas
        self.load_data_numpy_free()
    
    def safe_literal_eval(self, text):
        """Safely evaluate string representations of lists"""
        if not text or text == 'nan' or str(text).lower() == 'nan':
            return []
        try:
            if isinstance(text, str) and text.startswith('[') and text.endswith(']'):
                return ast.literal_eval(text)
            elif isinstance(text, str):
                return [item.strip() for item in text.split(',') if item.strip()]
            elif isinstance(text, list):
                return text
            else:
                return []
        except:
            return []
    
    def safe_int(self, value):
        """Safely convert value to int"""
        try:
            return int(float(value))
        except:
            return 0
    
    def load_data_numpy_free(self):
        """Load all data without using pandas or numpy"""
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Load PyTorch embeddings
        status_text.text("📥 Loading PyTorch embeddings...")
        embeddings_file = 'torch_recipe_embeddings_231630.pt'
        if os.path.exists(embeddings_file):
            self.recipe_embeddings = torch.load(embeddings_file, map_location=self.device)
            st.success(f"✅ Loaded {self.recipe_embeddings.shape[0]} embeddings")
            progress_bar.progress(25)
        else:
            st.error(f"❌ PyTorch embeddings not found: {embeddings_file}")
            self.recipe_embeddings = None
            self.is_ready = False
            return
        
        # Load recipes directly from CSV (numpy-free approach)
        status_text.text("📊 Loading recipes from CSV...")
        self.load_from_csv_direct(progress_bar, status_text)
        
        # Load statistics
        status_text.text("📈 Loading statistics...")
        stats_file = 'recipe_statistics_231630.pkl'
        try:
            if os.path.exists(stats_file):
                with open(stats_file, 'rb') as f:
                    self.recipe_stats = pickle.load(f)
                st.success(f"✅ Loaded statistics for {len(self.recipe_stats)} recipes")
            else:
                st.info("📈 Creating default statistics...")
                self.recipe_stats = {}
                for recipe in self.recipes:
                    self.recipe_stats[recipe['id']] = (4.0, 10, 5)
        except Exception as e:
            st.warning(f"⚠️ Statistics loading failed: {e}")
            self.recipe_stats = {}
            for recipe in self.recipes:
                self.recipe_stats[recipe['id']] = (4.0, 10, 5)
        
        progress_bar.progress(75)
        
        # Load scores
        status_text.text("⭐ Loading popularity scores...")
        scores_file = 'recipe_scores_231630.pkl'
        try:
            if os.path.exists(scores_file):
                with open(scores_file, 'rb') as f:
                    self.recipe_scores = pickle.load(f)
                st.success(f"✅ Loaded scores for {len(self.recipe_scores)} recipes")
            else:
                st.info("⭐ Creating default scores...")
                self.recipe_scores = {}
                for recipe in self.recipes:
                    self.recipe_scores[recipe['id']] = 0.5
        except Exception as e:
            st.warning(f"⚠️ Scores loading failed: {e}")
            self.recipe_scores = {}
            for recipe in self.recipes:
                self.recipe_scores[recipe['id']] = 0.5
        
        progress_bar.progress(100)
        
        # Check if we have everything
        self.is_ready = all([
            self.recipe_embeddings is not None,
            len(self.recipes) > 0,
            len(self.recipe_stats) > 0,
            len(self.recipe_scores) > 0
        ])
        
        if self.is_ready:
            self.fix_recipe_id_mismatches()
            status_text.text("🎯 All data loaded successfully!")
            st.success(f"🎉 System ready! Embeddings: {self.recipe_embeddings.shape[0]}, Recipes: {len(self.recipes)}")
        else:
            st.error("⚠️ Some data missing - search may be limited")
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
    
    def load_from_csv_direct(self, progress_bar, status_text):
        """Load recipes directly from CSV with exact filtering logic"""
        status_text.text("📊 Processing recipe dataset...")
        self.recipes = []
        
        if os.path.exists('RAW_recipes.csv'):
            valid_recipes = []
            
            with open('RAW_recipes.csv', 'r', encoding='utf-8') as file:
                csv_reader = csv.DictReader(file)
                
                for row_idx, row in enumerate(csv_reader):
                    try:
                        # Apply same filtering logic as embeddings
                        name = row.get('name', '')
                        if not name or str(name).lower().strip() in ['', 'nan', 'unknown recipe']:
                            continue
                        name = str(name).lower().strip()
                        
                        tags = self.safe_literal_eval(row.get('tags', '[]'))
                        ingredients = self.safe_literal_eval(row.get('ingredients', '[]'))
                        
                        # Filtering conditions
                        if not tags or len(tags) == 0:
                            continue
                        if not ingredients or len(ingredients) == 0:
                            continue
                        if len(name) == 0 or name == 'unknown recipe':
                            continue
                        
                        recipe = {
                            'id': int(row.get('id', row_idx)),
                            'name': name,
                            'minutes': self.safe_int(row.get('minutes', 0)),
                            'tags': tags,
                            'ingredients': ingredients,
                            'n_steps': self.safe_int(row.get('n_steps', 0)),
                            'description': str(row.get('description', '')).strip()
                        }
                        
                        valid_recipes.append(recipe)
                        
                        # Update progress every 10k recipes
                        if len(valid_recipes) % 10000 == 0:
                            progress_bar.progress(min(75, 25 + (len(valid_recipes) / 231630) * 50))
                            status_text.text(f"🔄 Processed {len(valid_recipes)} valid recipes...")
                        
                        # Stop at expected number
                        if len(valid_recipes) >= 231630:
                            break
                            
                    except Exception as e:
                        continue
            
            self.recipes = valid_recipes
            st.success(f"✅ Processed {len(self.recipes)} recipes")
        else:
            st.error("❌ RAW_recipes.csv not found")
            self.recipes = []
    
    def fix_recipe_id_mismatches(self):
        """Filter statistics and scores to match loaded recipes"""
        loaded_recipe_ids = set(recipe['id'] for recipe in self.recipes)
        
        original_stats = len(self.recipe_stats)
        self.recipe_stats = {
            recipe_id: stats for recipe_id, stats in self.recipe_stats.items()
            if recipe_id in loaded_recipe_ids
        }
        
        original_scores = len(self.recipe_scores)
        self.recipe_scores = {
            recipe_id: score for recipe_id, score in self.recipe_scores.items()
            if recipe_id in loaded_recipe_ids
        }
        
        st.info(f"🔧 Aligned data: Stats {original_stats}→{len(self.recipe_stats)}, Scores {original_scores}→{len(self.recipe_scores)}")
    
    def search_recipes(self, query: str, num_results: int = 5, min_rating: float = 3.0) -> List[Dict]:
        """Search for recipes and return results"""
        if not self.is_ready:
            return []
        
        if not query.strip():
            return []
        
        try:
            # Tokenize query
            inputs = self.tokenizer(
                query,
                return_tensors='pt',
                truncation=True,
                max_length=128,
                padding='max_length'
            ).to(self.device)
            
            # Get query embedding
            with torch.no_grad():
                outputs = self.model(**inputs)
                query_embedding = outputs.last_hidden_state[:, 0, :].cpu().flatten()
            
            # Calculate similarities
            recipe_embeddings_normalized = torch.nn.functional.normalize(self.recipe_embeddings, p=2, dim=1)
            query_embedding_normalized = torch.nn.functional.normalize(query_embedding.unsqueeze(0), p=2, dim=1)
            similarities = torch.mm(recipe_embeddings_normalized, query_embedding_normalized.t()).flatten()
            
            # Get top results
            top_indices = torch.argsort(similarities, descending=True)[:num_results * 3]
            
            results = []
            for idx in top_indices:
                if len(results) >= num_results:
                    break
                
                embedding_idx = idx.item()
                if embedding_idx < len(self.recipes):
                    recipe = self.recipes[embedding_idx]
                    recipe_id = recipe['id']
                    
                    if recipe_id in self.recipe_stats:
                        avg_rating, num_ratings, unique_users = self.recipe_stats[recipe_id]
                        
                        if avg_rating >= min_rating:
                            similarity_score = similarities[idx].item()
                            popularity_score = self.recipe_scores.get(recipe_id, 0.0)
                            combined_score = 0.7 * similarity_score + 0.3 * popularity_score
                            
                            results.append({
                                'name': recipe['name'],
                                'ingredients': recipe['ingredients'][:8] if isinstance(recipe['ingredients'], list) else [],
                                'tags': recipe['tags'][:6] if isinstance(recipe['tags'], list) else [],
                                'minutes': recipe.get('minutes', 0),
                                'n_steps': recipe.get('n_steps', 0),
                                'similarity_score': similarity_score,
                                'popularity_score': popularity_score,
                                'combined_score': combined_score,
                                'avg_rating': avg_rating,
                                'num_ratings': num_ratings,
                                'recipe_id': recipe_id
                            })
            
            return results
            
        except Exception as e:
            st.error(f"Search error: {str(e)}")
            return []

# Initialize the search system
@st.cache_resource
def load_search_system():
    return GoogleDriveRecipeSearch()

# Main app
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🍽️ Group 5 Pattern Recognition Project</h1>
        <h3>Advanced Recipe Recommendation using Semantic Search</h3>
        <p>🌐 Powered by Google Drive & Streamlit Cloud</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize search system
    if 'search_system' not in st.session_state:
        with st.spinner("🔄 Initializing recipe search system..."):
            st.session_state.search_system = load_search_system()
    
    search_system = st.session_state.search_system
    
    # Only show the app if system is ready
    if not search_system.is_ready:
        st.error("❌ System not ready. Please check the error messages above.")
        st.info("💡 Make sure all Google Drive file IDs are correctly configured.")
        return
    
    # Sidebar
    st.sidebar.header("🔍 Search Options")
    
    # Query input
    query = st.sidebar.text_input(
        "Search for recipes:",
        placeholder="e.g., 'chicken pasta', 'vegetarian salad', 'chocolate dessert'",
        help="Enter keywords to search for recipes"
    )
    
    # Search parameters
    num_results = st.sidebar.slider("Number of results", 1, 15, 5)
    min_rating = st.sidebar.slider("Minimum rating", 1.0, 5.0, 3.0, 0.1)
    
    # Example buttons
    st.sidebar.markdown("**💡 Quick Examples:**")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🍗 Chicken Pasta"):
            query = "chicken pasta"
        if st.button("🥩 Beef Steak"):
            query = "beef steak"
    
    with col2:
        if st.button("🥗 Healthy Salad"):
            query = "healthy salad"
        if st.button("🍫 Chocolate Dessert"):
            query = "chocolate dessert"
    
    # Search button
    search_clicked = st.sidebar.button("🔍 Search Recipes", type="primary")
    
    # System status
    with st.sidebar.expander("📊 System Status"):
        st.markdown(f"""
        <div class="download-status">
            ✅ <strong>System Ready!</strong><br>
            📊 {search_system.recipe_embeddings.shape[0]} recipes loaded<br>
            🧠 Device: {search_system.device}<br>
            🌐 Files loaded from Google Drive<br>
            🚀 Numpy-Free Implementation
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area
    if query and (search_clicked or query):
        # Perform search
        with st.spinner(f"🔍 Searching for '{query}'..."):
            start_time = time.time()
            results = search_system.search_recipes(query, num_results, min_rating)
            search_time = time.time() - start_time
        
        # Display results
        if results:
            st.markdown(f"## 🎯 Found {len(results)} recipes for '{query}'")
            st.markdown(f"⚡ Search completed in {search_time:.2f}s using trained BERT model")
            
            # Display each recipe
            for i, recipe in enumerate(results, 1):
                with st.container():
                    st.markdown(f"""
                    <div class="recipe-card">
                        <div class="recipe-title">{i}. {recipe['name']}</div>
                        
                        <div class="recipe-meta">
                            ⏱️ {recipe['minutes']} min | 
                            🔥 {recipe['n_steps']} steps | 
                            ⭐ {recipe['avg_rating']:.1f}/5.0 ({recipe['num_ratings']} ratings)
                        </div>
                        
                        <div style="margin: 8px 0;">
                            <span class="score-badge">Match: {recipe['similarity_score']:.1%}</span>
                            <span class="score-badge score-overall">Score: {recipe['combined_score']:.1%}</span>
                        </div>
                        
                        <div style="margin: 10px 0;">
                            {' '.join([f'<span class="recipe-tag">{tag}</span>' for tag in recipe['tags']])}
                        </div>
                        
                        <div style="margin: 10px 0;">
                            <strong>🥘 Ingredients:</strong><br>
                            {', '.join(recipe['ingredients'][:8])}
                            {'...' if len(recipe['ingredients']) > 8 else ''}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning(f"😔 No recipes found for '{query}' with minimum rating {min_rating}/5.0")
            st.info("💡 Try different keywords or lower the minimum rating filter.")
    
    else:
        # Welcome message
        st.markdown("""
        ## 👋 Welcome to Our Recipe Recommendation System
        
        This project demonstrates advanced machine learning techniques for intelligent recipe discovery.
        
        ### 🎯 Key Features:
        
        - 🧠 **Semantic Search** - Understands the meaning behind your queries
        - 🔍 **BERT Embeddings** - Uses a trained transformer model for recipe understanding
        - 📊 **Smart Scoring** - Combines similarity with popularity metrics
        - 🚀 **Fast Performance** - Search across 231,630 recipes in under 2 seconds
        - 🛠️ **Numpy-Free** - Robust implementation avoiding dependency conflicts
        - 🌐 **Cloud-Ready** - Files loaded from Google Drive for easy deployment
        
        ### 🔍 How to Use:
        
        1. **Enter your query** in the sidebar (e.g., "chicken pasta", "vegetarian salad")
        2. **Adjust search options** like number of results and minimum rating
        3. **Click Search** or try one of the quick example buttons
        4. **Browse results** with similarity scores and recipe details
        
        ### 📊 Dataset:
        
        - **231,630 recipes** with full ingredient and instruction data
        - **Custom-trained BERT model** for recipe-specific understanding
        - **Statistical analysis** of recipe ratings and popularity
        - **Semantic embeddings** for lightning-fast similarity matching
        
        **Ready to discover your next favorite recipe?** 🍽️
        """)

if __name__ == "__main__":
    main() 