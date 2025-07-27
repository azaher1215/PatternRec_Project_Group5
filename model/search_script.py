import torch
import numpy as np
from transformers import BertTokenizer, BertModel
import pickle
import json
import gdown
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import GOOGLE_DRIVE_FILES
import streamlit as st


class RecipeSearchSystem:
    
    def __init__(self, max_recipes=231630):
        try:
            # Load all the preprocessed files
            self.max_recipes = max_recipes
            file_paths = {
                'recipe_embeddings': GOOGLE_DRIVE_FILES['assets/nlp/advanced_recipe_embeddings_231630.npy'],
                'recipes_df': GOOGLE_DRIVE_FILES['assets/nlp/advanced_filtered_recipes_231630.pkl'],
                'recipe_stats': GOOGLE_DRIVE_FILES['assets/nlp/recipe_statistics_231630.pkl'],
                'model': GOOGLE_DRIVE_FILES['assets/nlp/tag_based_bert_model.pth']
                }
            output_path = "assets/nlp/"
            #download files from google drive
            self.ensure_files_exist(file_paths, output_path)
            # Set up device
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load tokenizer
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            
            # Load the trained model
            self.model = BertModel.from_pretrained('bert-base-uncased')
            self.model.load_state_dict(torch.load(f'{output_path}tag_based_bert_model.pth', map_location=self.device))
            
            self.model.to(self.device)
            self.model.eval()
            
            #load recipe embeddings
            self.recipe_embeddings = np.load(f'{output_path}advanced_recipe_embeddings_{self.max_recipes}.npy')
            #load recipes dataframe
            with open(f'{output_path}advanced_filtered_recipes_{self.max_recipes}.pkl', 'rb') as f:
                self.recipes_df = pickle.load(f)
            #load recipe statistics
            with open(f'{output_path}recipe_statistics_{self.max_recipes}.pkl', 'rb') as f:
                self.recipe_stats = pickle.load(f)
            self.is_ready = True
        except Exception as e:
            print(f"Error initializing search system: {e}")
            self.is_ready = False
    
    def ensure_files_exist(self, file_paths, output_path):
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        file_mapping = {
            'recipe_embeddings': f'advanced_recipe_embeddings_{self.max_recipes}.npy',
            'recipes_df': f'advanced_filtered_recipes_{self.max_recipes}.pkl',
            'recipe_stats': f'recipe_statistics_{self.max_recipes}.pkl',
            'model': f'tag_based_bert_model.pth'
        }
        
        for key, local_filename in file_mapping.items():
            local_path = os.path.join(output_path, local_filename)
            if not os.path.exists(local_path):
                print(f"Downloading {local_filename}...")
                gdown.download(file_paths[key], local_path, quiet=False, fuzzy=True)
                print(f"Downloaded {local_filename}")
            else:
                print(f"{local_filename} already exists, skipping download")
    
    def create_query_embedding(self, user_query):
        
        structured_query = f"anchor: {user_query.lower()}"
        
        # Tokenize the query
        tokenized_query = self.tokenizer(
            structured_query,
            return_tensors='pt',
            truncation=True,
            max_length=128,
            padding='max_length'
        )
        
        # Move to device
        tokenized_query = tokenized_query.to(self.device)
        
        # Get embedding from model
        with torch.no_grad():
            anchor_input_ids = tokenized_query['input_ids'].to(self.device)
            anchor_attention_mask = tokenized_query['attention_mask'].to(self.device)
            anchor_outputs = self.model(anchor_input_ids, anchor_attention_mask)
            # Get CLS token embedding
            anchor_embedding = anchor_outputs.last_hidden_state[:, 0, :]
            # Move to CPU and convert to numpy
            query_embedding_numpy = anchor_embedding.cpu().numpy().flatten()
        
        return query_embedding_numpy
    
    def calculate_similarities(self, query_embedding):
        similarities = []
        
        # Calculate cosine similarity for each recipe
        for i in range(len(self.recipe_embeddings)):
            recipe_embedding = self.recipe_embeddings[i]
            
            # Calculate cosine similarity
            #Cosine Similarity = (a · b) / (||a|| * ||b||)
            dot_product = np.dot(recipe_embedding, query_embedding)
            recipe_norm = np.linalg.norm(recipe_embedding)
            query_norm = np.linalg.norm(query_embedding)
            
            # Avoid division by zero
            if recipe_norm > 0 and query_norm > 0:
                similarity = dot_product / (recipe_norm * query_norm)
            else:
                similarity = 0.0
            
            similarities.append(similarity)
        
        return similarities
    
    def filter_recipes_by_quality(self, min_rating=3.0, min_num_ratings=5):
        #Get all indexes for recipes that meet the quality criteria the user chose
        filtered_recipe_indices = []
        
        for i in range(len(self.recipes_df)):
            recipe = self.recipes_df.iloc[i]
            recipe_id = recipe['id']
            
            if recipe_id in self.recipe_stats:
                avg_rating, num_ratings, _ = self.recipe_stats[recipe_id]
                
                if avg_rating >= min_rating and num_ratings >= min_num_ratings:
                    filtered_recipe_indices.append(i)
        
        return filtered_recipe_indices
    
    def rank_recipes_by_similarity_and_rating(self, similarities, recipe_indices):
        recipe_scores = []
        
        for recipe_index in recipe_indices:
            recipe = self.recipes_df.iloc[recipe_index]
            recipe_id = recipe['id']
            
            similarity_score = similarities[recipe_index]
            
            #if the recipe has no ratings we will assume it is a bad recipe to choose and set the ratio to 1.0
            if recipe_id in self.recipe_stats:
                avg_rating, _, _ = self.recipe_stats[recipe_id]
            else:
                avg_rating = 1.0  
            
            recipe_scores.append({
                'recipe_index': recipe_index,
                'recipe_id': recipe_id,
                'similarity_score': similarity_score,
                'avg_rating': avg_rating
            })
        
        return recipe_scores
    
    def create_recipe_result(self, recipe_index, scores_info):
        recipe = self.recipes_df.iloc[recipe_index]
        recipe_id = recipe['id']
        

        avg_rating, num_ratings, unique_users = self.recipe_stats[recipe_id]

        
        # Create result structure mapping
        result = {
            'recipe_id': int(recipe_id),
            'name': recipe['name'],
            'ingredients': recipe['ingredients'],
            'tags': recipe['tags'],
            'minutes': int(recipe['minutes']),
            'n_steps': int(recipe['n_steps']),
            'description': recipe.get('description', ''),
            'similarity_score': float(scores_info['similarity_score']),
            'avg_rating': float(avg_rating),
            'num_ratings': int(num_ratings),
            'unique_users': int(unique_users)
        }
        
        return result
    
    def search_recipes(self, user_query, top_k=5, min_rating=3.0, min_num_ratings=5):
                
        # Create embedding for user query
        query_embedding = self.create_query_embedding(user_query)
        
        # Calculate similarities between query and all recipes
        similarities = self.calculate_similarities(query_embedding)
        
        # Filter recipes by quality
        filtered_recipe_indices = self.filter_recipes_by_quality(min_rating, min_num_ratings)
        
        # Rank by semantic similarity and rating
        recipe_scores = self.rank_recipes_by_similarity_and_rating(similarities, filtered_recipe_indices)
        
        # Sort by semantic similarity, then by average rating
        recipe_scores.sort(key=lambda x: (x['similarity_score'], x['avg_rating']), reverse=True)
        
        # Get top results
        top_results = recipe_scores[:top_k]
        
        # Create result dictionaries
        final_results = []
        for score_info in top_results:
            recipe_result = self.create_recipe_result(score_info['recipe_index'], score_info)
            final_results.append(recipe_result)
        
        return final_results

@st.cache_resource
def search_for_recipes():
    return RecipeSearchSystem()


if __name__ == "__main__":
    
    search_system = RecipeSearchSystem()
    test_queries = [
        "chicken pasta italian quick dinner",
        "chocolate cake dessert brownie baked healthy",
        "healthy vegetarian salad tomato basil",
        "quick easy dinner",
        "beef steak",
        "beef pasta",
        "beef"
    ]
    
    for query in test_queries:
        print(f"Testing query: '{query}'")
        
        results = search_system.search_recipes(
            user_query=query,
            top_k=3,
            min_rating=3.5,
            min_num_ratings=10
        )
        
        print (results)
    print("Recipe search system testing complete!") 