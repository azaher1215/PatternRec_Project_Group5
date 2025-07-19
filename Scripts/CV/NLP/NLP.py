import pandas as pd
from ast import literal_eval
from transformers import BertTokenizer, BertModel
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch
import os
import numpy as np
from sklearn.model_selection import train_test_split
import re
from collections import defaultdict, Counter
import random

def clean_text(text):
  #helper function to clean the text from whitespace, double spaces
  # converts to lowecase and checks if the text is a string first to avoid errors
  if not isinstance(text, str):
    return ''
  text = text.lower()
  text = ' '.join(text.split())
  return text.strip()

def create_advanced_pair_data(recipes_df: pd.DataFrame, interactions_df: pd.DataFrame = None, 
                            num_pairs: int = 15000, use_quality_filtering: bool = True):
    """
    Advanced preprocessing function that creates high-quality anchor-positive-negative triplets
    optimized for BERT training on recipe data.
    
    Features:
    - Semantic tag categorization (cuisine, course, dietary, etc.)
    - Ingredient prioritization and cleaning
    - Quality-based recipe filtering using interaction data
    - Hard negative mining with semantic consistency
    - Structured text representation for better BERT understanding
    - Balanced query types (ingredient-focused, style-focused, mixed)
    
    Args:
        recipes_df: DataFrame with recipe data
        interactions_df: Optional DataFrame with user interactions for quality filtering
        num_pairs: Number of training pairs to generate
        use_quality_filtering: Whether to use interaction data for quality filtering
    
    Returns:
        DataFrame with columns ['anchor', 'positive', 'negative', 'query_type']
    """
    
    print(f"Creating {num_pairs} advanced training pairs...")
    
    # Tag categorization for semantic understanding
    tag_categories = {
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
    
    # Ingredient groupings for prioritization
    ingredient_groups = {
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
    
    def categorize_recipe_tags(recipe_tags):
        """Categorize recipe tags into semantic groups"""
        categorized = defaultdict(list)
        for tag in recipe_tags:
            tag_lower = tag.lower()
            for category, keywords in tag_categories.items():
                if any(keyword in tag_lower for keyword in keywords):
                    categorized[category].append(tag)
        return dict(categorized)
    
    def extract_main_ingredients(ingredients):
        """Extract and prioritize main ingredients"""
        cleaned_ingredients = []
        
        for ingredient in ingredients:
            # Remove common quantity words and modifiers
            cleaned = re.sub(r'\b(fresh|dried|chopped|minced|sliced|diced|ground|large|small|medium)\b', 
                           '', ingredient.lower()).strip()
            cleaned = re.sub(r'\d+|cup|cups|tablespoon|tablespoons|teaspoon|teaspoons|pound|pounds|ounce|ounces', 
                           '', cleaned).strip()
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            
            if cleaned and len(cleaned) > 2:
                cleaned_ingredients.append(cleaned)
        
        # Prioritize by ingredient groups
        prioritized = []
        
        # First, add proteins
        for ingredient in cleaned_ingredients:
            if any(protein in ingredient for protein in ingredient_groups['proteins']):
                prioritized.append(ingredient)
        
        # Then add vegetables and other components
        for group in ['vegetables', 'grains_starches', 'dairy']:
            for ingredient in cleaned_ingredients:
                if (ingredient not in prioritized and 
                    any(item in ingredient for item in ingredient_groups[group])):
                    prioritized.append(ingredient)
        
        # Add remaining ingredients
        for ingredient in cleaned_ingredients:
            if ingredient not in prioritized:
                prioritized.append(ingredient)
        
        return prioritized
    
    def create_structured_recipe_text(recipe):
        """Create optimized structured text representation"""
        # Extract and categorize tags
        categorized_tags = categorize_recipe_tags(recipe['tags'])
        
        # Priority order for tag categories
        priority_order = ['main_ingredient', 'cuisine', 'course', 'dietary', 'cooking_method']
        
        # Build structured tag text
        structured_tags = []
        for category in priority_order:
            if category in categorized_tags:
                structured_tags.extend(categorized_tags[category][:2])  # Max 2 per category
        
        # Add remaining important tags
        remaining_tags = [tag for tag in recipe['tags'] 
                         if tag not in structured_tags and 
                         any(keyword in tag.lower() for keyword in 
                             ['easy', 'quick', 'healthy', 'spicy', 'sweet'])]
        structured_tags.extend(remaining_tags[:3])
        
        # Process ingredients with grouping
        main_ingredients = extract_main_ingredients(recipe['ingredients'])
        
        # Create final structured text
        ingredients_text = ', '.join(main_ingredients[:8])
        tags_text = ', '.join(structured_tags[:10])
        
        # Include recipe name for context
        recipe_name = recipe['name'].replace(' ', ' ').strip()
        
        structured_text = f"Recipe: {recipe_name}. Ingredients: {ingredients_text}. Style: {tags_text}"
        
        return structured_text
    
    def create_semantic_query_from_recipe(recipe, query_type='mixed'):
        """Create semantic queries from recipe data"""
        categorized_tags = categorize_recipe_tags(recipe['tags'])
        main_ingredients = extract_main_ingredients(recipe['ingredients'])
        
        if query_type == 'ingredient_focused':
            # Create query focused on main ingredients
            selected_ingredients = main_ingredients[:3]
            selected_tags = []
            if 'cuisine' in categorized_tags:
                selected_tags.extend(categorized_tags['cuisine'][:1])
            if 'course' in categorized_tags:
                selected_tags.extend(categorized_tags['course'][:1])
            
            query = ' '.join(selected_ingredients + selected_tags)
            
        elif query_type == 'style_focused':
            # Create query focused on cooking style and dietary preferences
            selected_tags = []
            for category in ['cuisine', 'dietary', 'cooking_method', 'difficulty']:
                if category in categorized_tags:
                    selected_tags.extend(categorized_tags[category][:1])
            
            # Add one main ingredient for context
            if main_ingredients:
                selected_tags.append(main_ingredients[0])
            
            query = ' '.join(selected_tags)
            
        else:  # mixed
            # Balanced query with ingredients and style
            selected_ingredients = main_ingredients[:2]
            selected_tags = []
            
            for category in ['cuisine', 'course', 'dietary']:
                if category in categorized_tags:
                    selected_tags.extend(categorized_tags[category][:1])
            
            query = ' '.join(selected_ingredients + selected_tags)
        
        return query.lower().strip()
    
    def is_semantically_consistent_negative(query, positive_recipe, negative_recipe):
        """Check if negative recipe is semantically consistent"""
        query_lower = query.lower()
        neg_tags = ' '.join(negative_recipe['tags']).lower()
        neg_ingredients = ' '.join(negative_recipe['ingredients']).lower()
        
        # Dietary restriction consistency
        if any(diet in query_lower for diet in ['vegetarian', 'vegan']):
            meat_keywords = ['chicken', 'beef', 'pork', 'meat', 'fish', 'seafood', 'turkey', 'lamb']
            if any(meat in neg_ingredients for meat in meat_keywords):
                return False
        
        # Cuisine consistency (don't mix completely different cuisines)
        query_cuisines = [c for c in tag_categories['cuisine'] if c in query_lower]
        if query_cuisines:
            if not any(cuisine in neg_tags for cuisine in query_cuisines):
                return False
        
        # Course consistency (breakfast items shouldn't match dinner queries)
        if 'breakfast' in query_lower and 'dinner' in neg_tags:
            return False
        if 'dessert' in query_lower and 'main-dish' in neg_tags:
            return False
        
        # Main ingredient consistency
        query_ingredients = [ing for ing in ingredient_groups['proteins'] if ing in query_lower]
        if query_ingredients:
            if not any(ing in neg_ingredients for ing in query_ingredients):
                return False
        
        return True
    
    def mine_hard_negatives(query, positive_recipe, recipes_df, num_candidates=50):
        """Mine hard negative examples with semantic consistency checks"""
        query_words = set(query.lower().split())
        positive_categorized = categorize_recipe_tags(positive_recipe['tags'])
        positive_ingredients = set(ing.lower() for ing in positive_recipe['ingredients'])
        
        candidates = []
        
        for _, recipe in recipes_df.sample(n=min(num_candidates, len(recipes_df))).iterrows():
            if recipe['id'] == positive_recipe['id']:
                continue
            
            recipe_categorized = categorize_recipe_tags(recipe['tags'])
            recipe_ingredients = set(ing.lower() for ing in recipe['ingredients'])
            
            # Calculate semantic distance
            tag_overlap = 0
            for category in positive_categorized:
                if category in recipe_categorized:
                    tag_overlap += len(set(positive_categorized[category]).intersection(set(recipe_categorized[category])))
            
            ingredient_overlap = len(positive_ingredients.intersection(recipe_ingredients))
            query_overlap = len(query_words.intersection(
                set(' '.join(recipe['tags']).lower().split()) |
                set(' '.join(recipe['ingredients']).lower().split())
            ))
            
            # Score for hard negative (we want some similarity but not too much)
            negative_score = tag_overlap * 0.5 + ingredient_overlap * 0.3 + query_overlap * 0.2
            
            # Apply semantic consistency filters
            if is_semantically_consistent_negative(query, positive_recipe, recipe):
                candidates.append((recipe, negative_score))
        
        if not candidates:
            # Fallback to random selection
            return recipes_df.sample(n=1).iloc[0]
        
        # Sort by score and select from middle range (not too similar, not too different)
        candidates.sort(key=lambda x: x[1])
        middle_range = candidates[len(candidates)//4:3*len(candidates)//4] or candidates
        
        return random.choice(middle_range)[0]
    
    # Quality filtering using interactions data
    working_recipes = recipes_df.copy()
    
    if use_quality_filtering and interactions_df is not None:
        print("Applying quality filtering based on user interactions...")
        
        # Calculate recipe statistics
        recipe_stats = interactions_df.groupby('recipe_id').agg({
            'rating': ['mean', 'count', 'std'],
            'user_id': 'nunique'
        }).reset_index()
        
        recipe_stats.columns = ['id', 'avg_rating', 'num_ratings', 'rating_std', 'unique_users']
        
        # Merge with recipes and filter for quality
        recipes_with_quality = working_recipes.merge(recipe_stats, on='id', how='left')
        recipes_with_quality = recipes_with_quality.fillna({
            'avg_rating': 3.0, 'num_ratings': 1, 'rating_std': 1.0, 'unique_users': 1
        })
        
        # Filter high-quality recipes (rating >= 3.5, at least 5 ratings)
        quality_recipes = recipes_with_quality[
            (recipes_with_quality['avg_rating'] >= 3.5) & 
            (recipes_with_quality['num_ratings'] >= 5)
        ]
        
        if len(quality_recipes) > 1000:
            working_recipes = quality_recipes
            print(f"Using {len(working_recipes)} high-quality recipes for training")
        else:
            print(f"Only {len(quality_recipes)} quality recipes found, using all {len(working_recipes)} recipes")
    
    # Generate training pairs
    pairs = []
    query_types = ['ingredient_focused', 'style_focused', 'mixed']
    
    for i in range(num_pairs):
        # Select query type to ensure balance
        query_type = query_types[i % len(query_types)]
        
        # Select positive recipe
        positive_recipe = working_recipes.sample(n=1).iloc[0]
        
        # Create semantic query
        query = create_semantic_query_from_recipe(positive_recipe, query_type)
        
        # Skip if query is too short
        if len(query.split()) < 2:
            continue
        
        # Create structured positive text
        positive_text = create_structured_recipe_text(positive_recipe)
        
        # Mine hard negative
        negative_recipe = mine_hard_negatives(query, positive_recipe, working_recipes)
        negative_text = create_structured_recipe_text(negative_recipe)
        
        pairs.append({
            'anchor': query,
            'positive': positive_text,
            'negative': negative_text,
            'query_type': query_type
        })
        
        if (i + 1) % 1000 == 0:
            print(f"Created {i + 1}/{num_pairs} pairs...")
    
    pairs_df = pd.DataFrame(pairs)
    
    # Quality check
    print(f"\nAdvanced training pairs created:")
    print(f"Total pairs: {len(pairs_df)}")
    print(f"Average query length: {pairs_df['anchor'].str.len().mean():.1f} characters")
    print(f"Average positive text length: {pairs_df['positive'].str.len().mean():.1f} characters")
    print(f"Query type distribution:")
    print(pairs_df['query_type'].value_counts())
    
    # Show sample pairs
    print(f"\nSample training pairs:")
    for i, sample in pairs_df.head(3).iterrows():
        print(f"\nPair {i+1} ({sample['query_type']}):")
        print(f"  Query: {sample['anchor']}")
        print(f"  Positive: {sample['positive'][:80]}...")
        print(f"  Negative: {sample['negative'][:80]}...")
    
    return pairs_df

# Keep the original function for backward compatibility
def create_pair_data(recipes_df: pd.DataFrame):
  """Create positive and negative training pairs"""
  pair_data = []

  for i in range(2000):  # Increased number of pairs
    #select a random recipe for positive pair
    random_recipe = recipes_df.sample(n=1).iloc[0]

    #create query from recipe's tags and select up to 3 random tags
    pos_tags_selected = np.random.choice(random_recipe['tags'], size=min(5, len(random_recipe['tags'])), replace=False)
    query = ' '.join(pos_tags_selected)
    ingredients = ' '.join(random_recipe['ingredients'])
    tags = ' '.join(random_recipe['tags'])
    positive_recipe = ingredients + ' ' + tags

    #find negative recipe
    query_tags = set(query.split())
    negative_recipe = None
    max_attempts = 100
    attempts = 0

    while negative_recipe is None and attempts < max_attempts:
      random_neg_recipe = recipes_df.sample(n=1).iloc[0]
      random_recipe_tags = set(random_neg_recipe['tags'])

      #check if the negative recipe has minimal overlap with the query
      overlap = len(query_tags.intersection(random_recipe_tags))
      attempts += 1
      
      # More strict criteria for better negatives
      if overlap <= 2:  # Allow minimal overlap for harder negatives
        neg_ingredients = ' '.join(random_neg_recipe['ingredients'])
        neg_tags = ' '.join(random_neg_recipe['tags'])
        negative_recipe = neg_ingredients + ' ' + neg_tags
        print(f"Found negative recipe after {attempts} attempts. Overlap: {overlap}")
        break
    
    # If we found a negative recipe, add the pair
    if negative_recipe is not None:
      pair_data.append((query, positive_recipe, negative_recipe))
      print(f"Created pair {i+1}: Query='{query}', Overlap={overlap}")
    else:
      print(f"Could not find negative recipe for query '{query}' after {max_attempts} attempts")
    

  return pd.DataFrame(pair_data, columns=['anchor', 'positive', 'negative'])
    
class pos_neg_pair_dataset(Dataset):
  def __init__(self, pair_data, tokenizer, max_length=512):
    self.pair_data = pair_data
    self.tokenizer = tokenizer
    self.max_length = max_length

  def __len__(self):
    return len(self.pair_data)

  def __getitem__(self, idx):
    
    anchor = self.tokenizer(
      self.pair_data.iloc[idx]['anchor'], 
      return_tensors='pt', 
      truncation=True, 
      max_length=self.max_length, 
      padding='max_length')
    positive = self.tokenizer(
        self.pair_data.iloc[idx]['positive'], 
        return_tensors='pt', 
        truncation=True, 
        max_length=self.max_length, 
        padding='max_length')
    negative = self.tokenizer(
        self.pair_data.iloc[idx]['negative'], 
        return_tensors='pt', 
        truncation=True, 
        max_length=self.max_length, 
        padding='max_length')

    return {
      'anchor_input_ids': anchor['input_ids'].squeeze(),
      'anchor_attention_mask': anchor['attention_mask'].squeeze(),
      'positive_input_ids': positive['input_ids'].squeeze(),
      'positive_attention_mask': positive['attention_mask'].squeeze(),
      'negative_input_ids': negative['input_ids'].squeeze(),
      'negative_attention_mask': negative['attention_mask'].squeeze()
    }

def evaluate_model(model, val_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    total_loss = 0
    criterion = nn.TripletMarginLoss(margin=1.0)
    with torch.no_grad():
        for batch in val_loader:
            anchor_input_ids = batch['anchor_input_ids'].to(device)
            anchor_attention_mask = batch['anchor_attention_mask'].to(device)
            positive_input_ids = batch['positive_input_ids'].to(device)
            positive_attention_mask = batch['positive_attention_mask'].to(device)
            negative_input_ids = batch['negative_input_ids'].to(device)
            negative_attention_mask = batch['negative_attention_mask'].to(device)

            # Forward pass - get raw BERT embeddings
            anchor_outputs = model(anchor_input_ids, anchor_attention_mask)
            positive_outputs = model(positive_input_ids, positive_attention_mask)
            negative_outputs = model(negative_input_ids, negative_attention_mask)
            
            # Extract [CLS] token embeddings
            anchor_emb = anchor_outputs.last_hidden_state[:, 0, :]
            positive_emb = positive_outputs.last_hidden_state[:, 0, :]
            negative_emb = negative_outputs.last_hidden_state[:, 0, :]
            
            # Calculate loss
            loss = criterion(anchor_emb, positive_emb, negative_emb)
            
            total_loss += loss.item()
            
    print(f"Average loss on validation set: {total_loss/len(val_loader):.4f}")


def train_model(train_loader, val_loader, num_epochs=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.to(device)
    criterion = nn.TripletMarginLoss(margin=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            anchor_input_ids = batch['anchor_input_ids'].to(device)
            anchor_attention_mask = batch['anchor_attention_mask'].to(device)
            positive_input_ids = batch['positive_input_ids'].to(device)
            positive_attention_mask = batch['positive_attention_mask'].to(device)
            negative_input_ids = batch['negative_input_ids'].to(device)
            negative_attention_mask = batch['negative_attention_mask'].to(device)

            # Forward pass - get raw BERT embeddings
            anchor_outputs = model(anchor_input_ids, anchor_attention_mask)
            positive_outputs = model(positive_input_ids, positive_attention_mask)
            negative_outputs = model(negative_input_ids, negative_attention_mask)
            
            # Extract [CLS] token embeddings
            anchor_emb = anchor_outputs.last_hidden_state[:, 0, :]
            positive_emb = positive_outputs.last_hidden_state[:, 0, :]
            negative_emb = negative_outputs.last_hidden_state[:, 0, :]
            
            # Calculate loss
            loss = criterion(anchor_emb, positive_emb, negative_emb)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f'Epoch {epoch+1}, Average Loss: {total_loss/len(train_loader):.4f}')
        
    return model

if __name__ == '__main__':
  # Choose which preprocessing method to use
  use_advanced_preprocessing = True  # Set to False to use original method
  
  if not os.path.exists('pair_data.parquet'):
    # Load and prepare the data
    print("Loading recipe data...")
    recipes_df = pd.read_csv('RAW_recipes.csv')
    
    # Clean the data
    recipes_df['name'] = recipes_df['name'].apply(clean_text)
    recipes_df['tags'] = recipes_df['tags'].apply(literal_eval)
    recipes_df['ingredients'] = recipes_df['ingredients'].apply(literal_eval)

    # Filter recipes with meaningful data
    recipes_df = recipes_df[recipes_df['tags'].str.len() > 0]
    
    if use_advanced_preprocessing:
        # Load interactions for quality filtering
        print("Loading interaction data for quality filtering...")
        interactions_df = pd.read_csv('RAW_interactions.csv')
        interactions_df = interactions_df.dropna(subset=['rating'])
        interactions_df['rating'] = pd.to_numeric(interactions_df['rating'], errors='coerce')
        interactions_df = interactions_df.dropna(subset=['rating'])
        
        # Create advanced training pairs
        pair_data = create_advanced_pair_data(
            recipes_df, 
            interactions_df, 
            num_pairs=15000,  # Increased from 2000
            use_quality_filtering=True
        )
    else:
        # Use original method
        pair_data = create_pair_data(recipes_df)

    # Save the pair data
    pair_data.to_parquet('pair_data.parquet', index=False)
    print('Data saved to pair_data.parquet')

  else:
    pair_data = pd.read_parquet('pair_data.parquet')
    print('Data loaded from pair_data.parquet')

  # Split data to training and validation (80% training, 20% validation)
  train_data, val_data = train_test_split(pair_data, test_size=0.2, random_state=42)

# initialize tokenizer and model
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  
  # Create the datasets with reduced max_length for better performance
  train_dataset = pos_neg_pair_dataset(train_data, tokenizer, max_length=128)
  val_dataset = pos_neg_pair_dataset(val_data, tokenizer, max_length=128)

  # Create dataloaders with smaller batch size for stability
  train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

  # Train model
  print("Starting training...")
  model = train_model(train_loader, val_loader, num_epochs=3)

  #evaluate the model
  print("Evaluating model...")
  evaluate_model(model, val_loader)

  # Save model
  torch.save(model.state_dict(), 'tag_based_bert_model.pth')
  print("Model saved to tag_based_bert_model.pth")