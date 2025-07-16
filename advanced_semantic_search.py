import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from ast import literal_eval
from typing import List, Dict, Tuple
import os
import re
from collections import defaultdict
import pickle

class AdvancedSemanticRecipeSearch:
    """
    Advanced semantic search system that matches the preprocessing format 
    used in the updated NLP.py training pipeline.
    
    This ensures perfect compatibility between training and inference.
    """
    
    def __init__(self, model_path: str = 'tag_based_bert_model.pth', 
                 recipes_path: str = 'RAW_recipes.csv',
                 interactions_path: str = 'RAW_interactions.csv',
                 max_recipes: int = 231637):
        """
        Initialize the advanced semantic search system
        
        Args:
            model_path: Path to the trained BERT model
            recipes_path: Path to the recipes CSV file
            interactions_path: Path to the interactions CSV file
            max_recipes: Maximum number of recipes to process
        """
        print("Initializing Advanced Semantic Recipe Search System...")
        print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
        print(f"Max recipes: {max_recipes}")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_recipes = max_recipes
        
        # Tag categorization (same as in NLP.py)
        self.tag_categories = {
            'cuisine': [
                'italian', 'chinese', 'mexican', 'indian', 'french', 'greek', 'thai', 
                'japanese', 'american', 'european', 'asian', 'mediterranean', 'spanish', 
                'german', 'korean', 'vietnamese', 'turkish', 'moroccan', 'lebanese'
            ],
            'course': [
                'main-dish', 'side-dishes', 'appetizers', 'desserts', 'breakfast', 
                'lunch', 'dinner', 'snacks', 'beverages', 'salads', 'soups'
            ],
            'main_ingredient': [
                'chicken', 'beef', 'pork', 'fish', 'seafood', 'vegetables', 'fruit', 
                'pasta', 'rice', 'cheese', 'chocolate', 'potato', 'lamb', 'turkey',
                'beans', 'nuts', 'eggs', 'tofu'
            ],
            'dietary': [
                'vegetarian', 'vegan', 'gluten-free', 'low-carb', 'healthy', 'low-fat', 
                'diabetic', 'dairy-free', 'keto', 'paleo', 'whole30'
            ],
            'cooking_method': [
                'oven', 'stove-top', 'no-cook', 'microwave', 'slow-cooker', 'grilling', 
                'baking', 'roasting', 'frying', 'steaming', 'braising'
            ],
            'difficulty': ['easy', 'beginner-cook', 'advanced', 'intermediate', 'quick'],
            'time': [
                '15-minutes-or-less', '30-minutes-or-less', '60-minutes-or-less', 
                '4-hours-or-less', 'weeknight'
            ],
            'occasion': [
                'holiday-event', 'christmas', 'thanksgiving', 'valentines-day', 
                'summer', 'winter', 'spring', 'fall', 'party', 'picnic'
            ]
        }
        
        # Ingredient groupings (same as in NLP.py)
        self.ingredient_groups = {
            'proteins': [
                'chicken', 'beef', 'pork', 'fish', 'salmon', 'tuna', 'shrimp', 'turkey',
                'lamb', 'bacon', 'ham', 'sausage', 'eggs', 'tofu', 'beans', 'lentils'
            ],
            'vegetables': [
                'onion', 'garlic', 'tomato', 'carrot', 'celery', 'pepper', 'mushroom',
                'spinach', 'broccoli', 'zucchini', 'potato', 'sweet potato'
            ],
            'grains_starches': [
                'rice', 'pasta', 'bread', 'flour', 'oats', 'quinoa', 'barley', 'noodles'
            ],
            'dairy': [
                'milk', 'butter', 'cheese', 'cream', 'yogurt', 'sour cream', 'cream cheese'
            ]
        }
        
        # Load the trained model
        self.model = BertModel.from_pretrained('bert-base-uncased')
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded trained model from {model_path}")
        else:
            print(f"Warning: Model file {model_path} not found. Using pre-trained BERT.")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Load and prepare recipes and interactions
        self.recipes_df = self._load_and_prepare_recipes(recipes_path)
        self.interactions_df = self._load_interactions(interactions_path)
        
        # Calculate recipe popularity scores
        self.recipe_scores = self._calculate_recipe_scores()
        
        # Pre-compute recipe statistics for faster search
        self.recipe_stats = self._compute_recipe_statistics()
        
        # Pre-compute recipe embeddings for faster search
        self.recipe_embeddings = self._compute_recipe_embeddings()
        
    def _load_and_prepare_recipes(self, recipes_path: str) -> pd.DataFrame:
        """Load and prepare recipes data (same format as NLP.py)"""
        recipes_df = pd.read_csv(recipes_path)
        
        # Clean and parse data (same as NLP.py)
        # Handle NaN values in name field
        recipes_df['name'] = recipes_df['name'].fillna('unknown recipe')
        recipes_df['name'] = recipes_df['name'].astype(str).str.lower().str.strip()
        
        # Handle NaN values in description field
        recipes_df['description'] = recipes_df['description'].fillna('')
        recipes_df['description'] = recipes_df['description'].astype(str)
        
        # Parse tags and ingredients, handling potential string conversion issues
        recipes_df['tags'] = recipes_df['tags'].apply(lambda x: literal_eval(x) if isinstance(x, str) else [])
        recipes_df['ingredients'] = recipes_df['ingredients'].apply(lambda x: literal_eval(x) if isinstance(x, str) else [])
        
        # Filter recipes with meaningful data (same as NLP.py)
        recipes_df = recipes_df[recipes_df['tags'].apply(len) > 0]
        recipes_df = recipes_df[recipes_df['ingredients'].apply(len) > 0]
        
        # Filter out recipes with invalid names
        recipes_df = recipes_df[recipes_df['name'].str.len() > 0]
        recipes_df = recipes_df[recipes_df['name'] != 'unknown recipe']
        
        # Limit to max_recipes for testing
        if len(recipes_df) > self.max_recipes:
            print(f"Limiting to {self.max_recipes} recipes for testing (from {len(recipes_df)} total)")
            recipes_df = recipes_df.head(self.max_recipes)
        
        return recipes_df
    
    def _categorize_recipe_tags(self, recipe_tags: List[str]) -> Dict[str, List[str]]:
        """Categorize recipe tags into semantic groups (same as NLP.py)"""
        categorized = defaultdict(list)
        for tag in recipe_tags:
            tag_lower = tag.lower()
            for category, keywords in self.tag_categories.items():
                if any(keyword in tag_lower for keyword in keywords):
                    categorized[category].append(tag)
        return dict(categorized)
    
    def _extract_main_ingredients(self, ingredients: List[str]) -> List[str]:
        """Extract and prioritize main ingredients (same as NLP.py)"""
        if not ingredients or not isinstance(ingredients, list):
            return []
            
        cleaned_ingredients = []
        
        for ingredient in ingredients:
            try:
                # Ensure ingredient is a string
                ingredient_str = str(ingredient) if ingredient is not None else ''
                if not ingredient_str or ingredient_str == 'nan':
                    continue
                    
                # Remove common quantity words and modifiers
                cleaned = re.sub(r'\b(fresh|dried|chopped|minced|sliced|diced|ground|large|small|medium)\b', 
                               '', ingredient_str.lower()).strip()
                cleaned = re.sub(r'\d+|cup|cups|tablespoon|tablespoons|teaspoon|teaspoons|pound|pounds|ounce|ounces', 
                               '', cleaned).strip()
                cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                
                if cleaned and len(cleaned) > 2:
                    cleaned_ingredients.append(cleaned)
            except Exception as e:
                # Skip problematic ingredients
                continue
        
        # Prioritize by ingredient groups
        prioritized = []
        
        # First, add proteins
        for ingredient in cleaned_ingredients:
            if any(protein in ingredient for protein in self.ingredient_groups['proteins']):
                prioritized.append(ingredient)
        
        # Then add vegetables and other components
        for group in ['vegetables', 'grains_starches', 'dairy']:
            for ingredient in cleaned_ingredients:
                if (ingredient not in prioritized and 
                    any(item in ingredient for item in self.ingredient_groups[group])):
                    prioritized.append(ingredient)
        
        # Add remaining ingredients
        for ingredient in cleaned_ingredients:
            if ingredient not in prioritized:
                prioritized.append(ingredient)
        
        return prioritized
    
    def _create_structured_recipe_text(self, recipe: pd.Series) -> str:
        """Create structured recipe text (EXACT same format as NLP.py training)"""
        try:
            # Extract and categorize tags
            recipe_tags = recipe['tags'] if isinstance(recipe['tags'], list) else []
            categorized_tags = self._categorize_recipe_tags(recipe_tags)
            
            # Priority order for tag categories
            priority_order = ['main_ingredient', 'cuisine', 'course', 'dietary', 'cooking_method']
            
            # Build structured tag text
            structured_tags = []
            for category in priority_order:
                if category in categorized_tags:
                    structured_tags.extend(categorized_tags[category][:2])  # Max 2 per category
            
            # Add remaining important tags
            remaining_tags = [tag for tag in recipe_tags 
                             if tag not in structured_tags and 
                             any(keyword in str(tag).lower() for keyword in 
                                 ['easy', 'quick', 'healthy', 'spicy', 'sweet'])]
            structured_tags.extend(remaining_tags[:3])
            
            # Process ingredients with grouping
            recipe_ingredients = recipe['ingredients'] if isinstance(recipe['ingredients'], list) else []
            main_ingredients = self._extract_main_ingredients(recipe_ingredients)
            
            # Create final structured text (EXACT same format as NLP.py)
            ingredients_text = ', '.join(main_ingredients[:8])
            tags_text = ', '.join(structured_tags[:10])
            
            # Include recipe name for context - ensure it's a string
            recipe_name = str(recipe['name']).replace('  ', ' ').strip()
            if not recipe_name or recipe_name == 'nan':
                recipe_name = 'recipe'
            
            structured_text = f"Recipe: {recipe_name}. Ingredients: {ingredients_text}. Style: {tags_text}"
            
            return structured_text
            
        except Exception as e:
            # Fallback for any edge cases
            print(f"Warning: Error processing recipe {recipe.get('id', 'unknown')}: {e}")
            recipe_name = str(recipe.get('name', 'recipe')).strip()
            return f"Recipe: {recipe_name}. Ingredients: . Style: ."
    
    def _load_interactions(self, interactions_path: str) -> pd.DataFrame:
        """Load and prepare interactions data"""
        interactions_df = pd.read_csv(interactions_path)
        
        # Clean interactions data
        interactions_df = interactions_df.dropna(subset=['rating'])
        interactions_df['rating'] = pd.to_numeric(interactions_df['rating'], errors='coerce')
        interactions_df = interactions_df.dropna(subset=['rating'])
        interactions_df['review'] = interactions_df['review'].fillna('')
        
        return interactions_df
    
    def _calculate_recipe_scores(self) -> Dict[int, float]:
        """Calculate popularity and quality scores for each recipe"""
        scores_path = f'recipe_scores_{self.max_recipes}.pkl'
        
        # Check if scores already exist
        if os.path.exists(scores_path):
            print("Loading pre-computed recipe scores...")
            with open(scores_path, 'rb') as f:
                recipe_scores = pickle.load(f)
            print(f"Loaded scores for {len(recipe_scores)} recipes")
            return recipe_scores
        
        print("Calculating recipe popularity scores...")
        
        # Group interactions by recipe_id
        recipe_stats = self.interactions_df.groupby('recipe_id').agg({
            'rating': ['mean', 'count', 'std'],
            'user_id': 'nunique'
        }).reset_index()
        
        recipe_stats.columns = ['recipe_id', 'avg_rating', 'num_ratings', 'rating_std', 'unique_users']
        
        # Calculate weighted score
        recipe_stats['weighted_score'] = (
            recipe_stats['avg_rating'] * 
            np.log1p(recipe_stats['num_ratings']) *
            (1 + recipe_stats['unique_users'] / 100)
        )
        
        # Normalize scores to 0-1 range
        recipe_stats['normalized_score'] = (
            recipe_stats['weighted_score'] - recipe_stats['weighted_score'].min()
        ) / (recipe_stats['weighted_score'].max() - recipe_stats['weighted_score'].min())
        
        # Create dictionary mapping recipe_id to score
        recipe_scores = dict(zip(recipe_stats['recipe_id'], recipe_stats['normalized_score']))
        
        # Save scores for future use
        print(f"Saving scores for {len(recipe_scores)} recipes...")
        with open(scores_path, 'wb') as f:
            pickle.dump(recipe_scores, f)
        print("Recipe scores saved successfully!")
        
        return recipe_scores
    
    def _compute_recipe_statistics(self) -> Dict[int, Tuple[float, int, int]]:
        """Pre-compute recipe statistics for faster search"""
        stats_path = f'recipe_statistics_{self.max_recipes}.pkl'
        
        # Check if statistics already exist
        if os.path.exists(stats_path):
            print("Loading pre-computed recipe statistics...")
            with open(stats_path, 'rb') as f:
                recipe_stats = pickle.load(f)
            print(f"Loaded statistics for {len(recipe_stats)} recipes")
            return recipe_stats
        
        print("Computing recipe statistics...")
        
        # Group interactions by recipe_id
        stats_df = self.interactions_df.groupby('recipe_id').agg({
            'rating': ['mean', 'count'],
            'user_id': 'nunique'
        }).reset_index()
        
        stats_df.columns = ['recipe_id', 'avg_rating', 'num_ratings', 'unique_users']
        
        # Create dictionary for fast lookup
        recipe_stats = dict(zip(stats_df['recipe_id'], 
                               zip(stats_df['avg_rating'], 
                                   stats_df['num_ratings'], 
                                   stats_df['unique_users'])))
        
        # Save statistics for future use
        print(f"Saving statistics for {len(recipe_stats)} recipes...")
        with open(stats_path, 'wb') as f:
            pickle.dump(recipe_stats, f)
        print("Recipe statistics saved successfully!")
        
        return recipe_stats
    
    def _compute_recipe_embeddings(self) -> np.ndarray:
        """Pre-compute embeddings for all recipes using the trained model"""
        embeddings_path = f'advanced_recipe_embeddings_{self.max_recipes}.npy'
        filtered_recipes_path = f'advanced_filtered_recipes_{self.max_recipes}.pkl'
        
        # Check if embeddings already exist
        if os.path.exists(embeddings_path) and os.path.exists(filtered_recipes_path):
            print("Loading pre-computed recipe embeddings...")
            embeddings = np.load(embeddings_path)
            
            # Load the filtered recipes DataFrame
            with open(filtered_recipes_path, 'rb') as f:
                self.recipes_df = pickle.load(f)
            
            print(f"Loaded {len(embeddings)} recipe embeddings for {len(self.recipes_df)} recipes")
            return embeddings
        
        print("Computing recipe embeddings with trained model (this may take a while)...")
        embeddings = []
        valid_recipes = []
        
        with torch.no_grad():
            for idx, recipe in self.recipes_df.iterrows():
                try:
                    # Create structured recipe text (same format as training)
                    recipe_text = self._create_structured_recipe_text(recipe)
                    
                    # Skip empty or problematic recipe texts
                    if not recipe_text or len(recipe_text.strip()) < 10:
                        print(f"Skipping recipe {recipe.get('id', idx)} - insufficient text")
                        continue
                    
                    # Tokenize recipe text
                    inputs = self.tokenizer(
                        recipe_text,
                        return_tensors='pt',
                        truncation=True,
                        max_length=128,  # Same as training
                        padding='max_length'
                    ).to(self.device)
                    
                    # Get embedding using trained model
                    outputs = self.model(**inputs)
                    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # [CLS] token
                    embeddings.append(embedding.flatten())
                    
                    # Store the valid recipe data
                    valid_recipes.append(recipe.copy())
                    
                    # Progress indicator
                    if len(embeddings) % 1000 == 0:
                        print(f"Processed {len(embeddings)} recipes...")
                        
                except Exception as e:
                    print(f"Error processing recipe {recipe.get('id', idx)}: {e}")
                    continue
        
        if not embeddings:
            raise ValueError("No valid recipe embeddings could be computed!")
        
        embeddings = np.array(embeddings)
        
        # Create new DataFrame from valid recipes with clean indices
        valid_recipes_df = pd.DataFrame(valid_recipes).reset_index(drop=True)
        
        # Save embeddings and filtered recipes DataFrame for future use
        print(f"Saving {len(embeddings)} recipe embeddings...")
        np.save(embeddings_path, embeddings)
        
        with open(filtered_recipes_path, 'wb') as f:
            pickle.dump(valid_recipes_df, f)
        
        print("Embeddings and filtered recipes saved successfully!")
        
        # Update recipes_df to the filtered version
        self.recipes_df = valid_recipes_df
        print(f"Filtered to {len(self.recipes_df)} valid recipes")
        
        return embeddings
    
    def search_recipes(self, query: str, top_k: int = 5, 
                      min_rating: float = 3.0, 
                      min_ratings: int = 5,
                      use_popularity: bool = True,
                      semantic_weight: float = 0.9) -> List[Dict]:
        """
        Search for recipes based on user query using the trained model
        
        Args:
            query: User query (e.g., "chicken italian pasta")
            top_k: Number of top results to return
            min_rating: Minimum average rating required
            min_ratings: Minimum number of ratings required
            use_popularity: Whether to incorporate popularity scores
            semantic_weight: Weight for semantic similarity (0.0-1.0)
            
        Returns:
            List of dictionaries containing recipe information and similarity score
        """
        # Create structured query (same format as training)
        structured_query = f"Query: {query.lower()}"
        
        # Tokenize query
        query_inputs = self.tokenizer(
            structured_query,
            return_tensors='pt',
            truncation=True,
            max_length=128,  # Same as training
            padding='max_length'
        ).to(self.device)
        
        # Get query embedding using trained model
        with torch.no_grad():
            query_outputs = self.model(**query_inputs)
            query_embedding = query_outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
        
        # Calculate similarities
        similarities = np.dot(self.recipe_embeddings, query_embedding) / (
            np.linalg.norm(self.recipe_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Filter and rank recipes using pre-computed statistics
        filtered_results = []
        for df_idx, recipe in self.recipes_df.iterrows():
            recipe_id = recipe['id']
            
            # Get recipe statistics from pre-computed data
            if recipe_id in self.recipe_stats:
                avg_rating, num_ratings, unique_users = self.recipe_stats[recipe_id]
                
                # Apply filters
                if avg_rating >= min_rating and num_ratings >= min_ratings:
                    # Get similarity score using df_idx (which corresponds to embeddings array index)
                    if df_idx < len(similarities):
                        semantic_score = similarities[df_idx]
                        popularity_score = self.recipe_scores.get(recipe_id, 0.0)
                        
                        if use_popularity:
                            # Use configurable semantic weight
                            combined_score = semantic_weight * semantic_score + (1 - semantic_weight) * popularity_score
                        else:
                            combined_score = semantic_score
                        
                        filtered_results.append({
                            'recipe_id': int(recipe_id),
                            'name': recipe['name'],
                            'ingredients': recipe['ingredients'],
                            'tags': recipe['tags'],
                            'minutes': int(recipe['minutes']),
                            'n_steps': int(recipe['n_steps']),
                            'similarity_score': float(semantic_score),
                            'popularity_score': float(popularity_score),
                            'combined_score': float(combined_score),
                            'avg_rating': float(avg_rating),
                            'num_ratings': int(num_ratings),
                            'unique_users': int(unique_users),
                            'description': recipe.get('description', ''),
                            'recipe_text': self._create_structured_recipe_text(recipe)
                        })
                    else:
                        print(f"Warning: Index mismatch for recipe {recipe_id} at df_idx {df_idx}")
        
        # Sort by combined score and return top-k
        filtered_results.sort(key=lambda x: x['combined_score'], reverse=True)
        return filtered_results[:top_k]
    
    def search_by_tags(self, tags: List[str], top_k: int = 5, 
                      min_rating: float = 3.0, 
                      min_ratings: int = 5,
                      use_popularity: bool = True,
                      semantic_weight: float = 0.9) -> List[Dict]:
        """
        Search for recipes based on specific tags
        
        Args:
            tags: List of tags (e.g., ["chicken", "italian", "pasta"])
            top_k: Number of top results to return
            min_rating: Minimum average rating required
            min_ratings: Minimum number of ratings required
            use_popularity: Whether to incorporate popularity scores
            semantic_weight: Weight for semantic similarity (0.0-1.0)
            
        Returns:
            List of dictionaries containing recipe information
        """
        query = ' '.join(tags)
        return self.search_recipes(query, top_k, min_rating, min_ratings, use_popularity, semantic_weight)

    def force_recompute(self):
        """Force recomputation of embeddings and statistics"""
        print("Forcing recomputation of all cached data...")
        
        # Remove existing cache files
        cache_files = [
            f'advanced_recipe_embeddings_{self.max_recipes}.npy', 
            f'advanced_filtered_recipes_{self.max_recipes}.pkl',
            f'recipe_scores_{self.max_recipes}.pkl', 
            f'recipe_statistics_{self.max_recipes}.pkl'
        ]
        for file in cache_files:
            if os.path.exists(file):
                os.remove(file)
                print(f"Removed {file}")
        
        # Reload original recipes data
        self.recipes_df = self._load_and_prepare_recipes('RAW_recipes.csv')
        
        # Recompute everything
        self.recipe_scores = self._calculate_recipe_scores()
        self.recipe_stats = self._compute_recipe_statistics()
        self.recipe_embeddings = self._compute_recipe_embeddings()
        
        print("Recomputation complete!")

# Enhanced API endpoint functions
def get_recipe_recommendations(query: str, top_k: int = 5, 
                             min_rating: float = 3.0, 
                             min_ratings: int = 5,
                             use_popularity: bool = True,
                             semantic_weight: float = 0.9) -> List[Dict]:
    """
    API endpoint function for getting recipe recommendations
    
    Args:
        query: User query string
        top_k: Number of recommendations to return
        min_rating: Minimum average rating required
        min_ratings: Minimum number of ratings required
        use_popularity: Whether to incorporate popularity scores
        semantic_weight: Weight for semantic similarity (0.0-1.0)
        
    Returns:
        List of recipe recommendations with metadata and user feedback
    """
    # Initialize search system (this should be done once at startup)
    if not hasattr(get_recipe_recommendations, 'search_system'):
        get_recipe_recommendations.search_system = AdvancedSemanticRecipeSearch()
    
    # Perform search
    results = get_recipe_recommendations.search_system.search_recipes(
        query, top_k, min_rating, min_ratings, use_popularity, semantic_weight
    )
    
    return results

def get_recipe_recommendations_by_tags(tags: List[str], top_k: int = 5,
                                      min_rating: float = 3.0, 
                                      min_ratings: int = 5,
                                      use_popularity: bool = True,
                                      semantic_weight: float = 0.9) -> List[Dict]:
    """
    API endpoint function for getting recipe recommendations by tags
    
    Args:
        tags: List of tag strings
        top_k: Number of recommendations to return
        min_rating: Minimum average rating required
        min_ratings: Minimum number of ratings required
        use_popularity: Whether to incorporate popularity scores
        semantic_weight: Weight for semantic similarity (0.0-1.0)
        
    Returns:
        List of recipe recommendations with metadata and user feedback
    """
    # Initialize search system (this should be done once at startup)
    if not hasattr(get_recipe_recommendations_by_tags, 'search_system'):
        get_recipe_recommendations_by_tags.search_system = AdvancedSemanticRecipeSearch()
    
    # Perform search
    results = get_recipe_recommendations_by_tags.search_system.search_by_tags(
        tags, top_k, min_rating, min_ratings, use_popularity, semantic_weight
    )
    
    return results

# Example usage and testing
if __name__ == "__main__":
    # Test the advanced search system
    search_system = AdvancedSemanticRecipeSearch()
    
    # Test queries that should work well with the trained model
    test_queries = [
        "chicken italian pasta",
        "beef steak dinner",
        "vegetarian salad healthy",
        "dessert chocolate cake",
        "quick easy dinner"
    ]
    
    print("Testing advanced semantic search system with trained model...")
    print("=" * 80)
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        
        # Test with high semantic weight (focus on semantic similarity)
        results = search_system.search_recipes(
            query, 
            top_k=3, 
            min_rating=3.0,
            min_ratings=5,
            use_popularity=True,
            semantic_weight=0.95  # 95% semantic, 5% popularity
        )
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['name']}")
                print(f"   Combined Score: {result['combined_score']:.3f}")
                print(f"   Semantic Score: {result['similarity_score']:.3f}")
                print(f"   Popularity Score: {result['popularity_score']:.3f}")
                print(f"   Avg Rating: {result['avg_rating']:.1f} ({result['num_ratings']} ratings)")
                print(f"   Tags: {', '.join(result['tags'][:5])}")
                print(f"   Time: {result['minutes']} minutes")
                print()
        else:
            print("   No results found with current filters")
    
    print("Advanced search system ready for production use!") 