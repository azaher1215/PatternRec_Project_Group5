import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from ast import literal_eval
import re
import pickle
from datetime import datetime

def clean_text(text):
    #helper function to clean the text from whitespace, double spaces
    # converts to lowercase and checks if the text is a string first to avoid errors
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = ' '.join(text.split())
    return text.strip()

def setup_tag_categories():
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
    return tag_categories

def setup_ingredient_groups():
    
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
    return ingredient_groups

def load_and_clean_recipes(recipes_path):
    print(f"Loading recipes from {recipes_path}")
    
    # Load the CSV file
    recipes_df = pd.read_csv(recipes_path)
    
    # Clean the recipe names
    recipes_df['name'] = recipes_df['name'].fillna('unknown recipe').astype(str).apply(clean_text)
    
    # Update the dataframe
    recipes_df['description'] = recipes_df['description'].fillna('').astype(str).apply(clean_text)
    
    # cleaning tags and ingredients from string format
    recipes_df['tags'] = recipes_df['tags'].apply(literal_eval)
    recipes_df['ingredients'] = recipes_df['ingredients'].apply(literal_eval)
    
    # Filter out recipes with no tags or ingredients
    recipes_df = recipes_df[
        (recipes_df['tags'].str.len() > 0) & 
        (recipes_df['ingredients'].str.len() > 0) & 
        (recipes_df['name'].str.len() > 0) & 
        (recipes_df['name'] != 'unknown recipe')
    ].reset_index(drop=True)
    
    
    print(f"Final number of valid recipes: {len(recipes_df)}")
    return recipes_df

def categorize_recipe_tags(recipe_tags, tag_categories):
    categorized_tags = {}
    
    # Initialize empty lists for each category
    for category_name in tag_categories.keys():
        categorized_tags[category_name] = []
    
    # Check each tag
    for tag in recipe_tags:
        tag_lower = tag.lower()
        
        # Check each category
        for category_name in tag_categories.keys():
            category_keywords = tag_categories[category_name]
            
            # Check if any keyword matches this tag
            for keyword in category_keywords:
                if keyword in tag_lower:
                    categorized_tags[category_name].append(tag)
                    break
    
    return categorized_tags

def extract_main_ingredients(ingredients_list, ingredient_groups):
    if not ingredients_list or not isinstance(ingredients_list, list):
        return []
    
    # Clean each ingredient
    cleaned_ingredients = []
    
    for ingredient in ingredients_list:
        # Convert to string
        ingredient_string = str(ingredient) if ingredient is not None else ''
        if not ingredient_string or ingredient_string == 'nan':
            continue
        
        # Make lowercase
        cleaned_ingredient = ingredient_string.lower()
        
        # Remove common descriptor words
        words_to_remove = ['fresh', 'dried', 'chopped', 'minced', 'sliced', 'diced', 'ground', 'large', 'small', 'medium']
        for word in words_to_remove:
            cleaned_ingredient = cleaned_ingredient.replace(word, '')
        
        # Remove numbers
        cleaned_ingredient = re.sub(r'\d+', '', cleaned_ingredient)
        
        # Remove measurement words
        measurement_words = ['cup', 'cups', 'tablespoon', 'tablespoons', 'teaspoon', 'teaspoons', 'pound', 'pounds', 'ounce', 'ounces']
        for measurement in measurement_words:
            cleaned_ingredient = cleaned_ingredient.replace(measurement, '')
        
        # Clean up extra spaces
        cleaned_ingredient = re.sub(r'\s+', ' ', cleaned_ingredient).strip()
        
        # Only keep if it's long enough
        if cleaned_ingredient and len(cleaned_ingredient) > 2:
            cleaned_ingredients.append(cleaned_ingredient)

    
    # Put ingredients in order of importance
    ordered_ingredients = []
    
    # First, add proteins (most important)
    for ingredient in cleaned_ingredients:
        for protein in ingredient_groups['proteins']:
            if protein in ingredient:
                ordered_ingredients.append(ingredient)
                break
        
    
    # Then add vegetables, grains, and dairy
    other_groups = ['vegetables', 'grains_starches', 'dairy']
    for group_name in other_groups:
        for ingredient in cleaned_ingredients:
            if ingredient not in ordered_ingredients:
                for group_item in ingredient_groups[group_name]:
                    if group_item in ingredient:
                        ordered_ingredients.append(ingredient)
                        break
    
    # Finally, add any remaining ingredients
    for ingredient in cleaned_ingredients:
        if ingredient not in ordered_ingredients:
            ordered_ingredients.append(ingredient)
    
    return ordered_ingredients

def create_structured_recipe_text(recipe, tag_categories, ingredient_groups):
    # Get recipe tags and categorize them
    recipe_tags = recipe['tags'] if isinstance(recipe['tags'], list) else []
    categorized_tags = categorize_recipe_tags(recipe_tags, tag_categories)
    
    # Choose tags in priority order
    priority_categories = ['main_ingredient', 'cuisine', 'course', 'dietary', 'cooking_method']
    selected_tags = []
    
    for category in priority_categories:
        if category in categorized_tags:
            # Take up to 2 tags from each category
            category_tags = categorized_tags[category][:2]
            for tag in category_tags:
                selected_tags.append(tag)
    
    # Add some additional important tags
    important_keywords = ['easy', 'quick', 'healthy', 'spicy', 'sweet']
    remaining_tags = []
    
    for tag in recipe_tags:
        if tag not in selected_tags:  
            for keyword in important_keywords:
                if keyword in tag.lower():
                    remaining_tags.append(tag)
                    break
            
    
    # Add up to 3 remaining tags
    for i in range(min(3, len(remaining_tags))):
        selected_tags.append(remaining_tags[i])
    
    # Process ingredients
    recipe_ingredients = recipe['ingredients'] if isinstance(recipe['ingredients'], list) else []
    main_ingredients = extract_main_ingredients(recipe_ingredients, ingredient_groups)
    
    # Step 5: Create the final structured text
    # Join first 8 ingredients
    ingredients_text = ', '.join(main_ingredients[:8])
    
    # Join first 10 tags
    tags_text = ', '.join(selected_tags[:10])
    
    # Get recipe name
    recipe_name = str(recipe['name']).replace('  ', ' ').strip()
    
    # Create final structured text
    structured_text = f"Recipe: {recipe_name}. Ingredients: {ingredients_text}. Style: {tags_text}"
    
    return structured_text
        

def create_recipe_statistics(interactions_path='RAW_interactions.csv'):
    print("Creating recipe statistics")
        
    # Load interactions data
    interactions_df = pd.read_csv(interactions_path)    
    # Clean interactions data    
    interactions_df = interactions_df.dropna(subset=['rating'])
    # Convert ratings to numbers
    interactions_df['rating'] = pd.to_numeric(interactions_df['rating'], errors='coerce')
    
    # Remove rows where rating conversion failed
    interactions_df = interactions_df.dropna(subset=['rating'])
    
    print(f"Valid interactions after cleaning: {len(interactions_df)}")
    
    # Calculate statistics for each recipe
    recipe_stats = {}
    unique_recipe_ids = interactions_df['recipe_id'].unique()
    
    for recipe_id in unique_recipe_ids:
        # Get all interactions for this recipe
        recipe_interactions = interactions_df[interactions_df['recipe_id'] == recipe_id]
        # Calculate average rating
        ratings_list = recipe_interactions['rating'].tolist()
        average_rating = sum(ratings_list) / len(ratings_list)
        # Count number of ratings
        number_of_ratings = len(recipe_interactions)
        # Count unique users
        unique_users = recipe_interactions['user_id'].nunique()
        
        recipe_stats[recipe_id] = (average_rating, number_of_ratings, unique_users)
    
    print(f"Created statistics for {len(recipe_stats)} recipes")
    return recipe_stats

def create_recipe_embeddings(recipes_df, model, tokenizer, device, tag_categories, ingredient_groups):
    print("Creating recipe embeddings (this will take a long time)")
    
    recipe_embeddings_list = []
    valid_recipes_list = []
    
    # Process each recipe one by one
    for i in range(len(recipes_df)):
        recipe = recipes_df.iloc[i]
        
        try:
            # Create structured text for this recipe
            recipe_text = create_structured_recipe_text(recipe, tag_categories, ingredient_groups)
            
            # Tokenize the recipe text
            tokenized_input = tokenizer(
                recipe_text,
                return_tensors='pt',
                truncation=True,
                max_length=128,
                padding='max_length'
            )
            
            
            # Get embedding from model
            with torch.no_grad():
                tokenized_input = tokenized_input['input_ids'].to(device)
                tokenized_mask = tokenized_input['attention_mask'].to(device)
                model_outputs = model(tokenized_input, tokenized_mask)
                # Get CLS token embedding (first token)
                cls_embedding = model_outputs.last_hidden_state[:, 0, :]
                # Move to CPU and convert to numpy
                embedding_numpy = cls_embedding.cpu().numpy().flatten()
            
            # Store the embedding and recipe
            recipe_embeddings_list.append(embedding_numpy)
            valid_recipes_list.append(recipe.copy())
            
            # Show progress every 1000 recipes
            if len(recipe_embeddings_list) % 1000 == 0:
                print(f"Processed {len(recipe_embeddings_list)} recipes")
                
        except Exception as e:
            print(f"Error processing recipe {recipe.get('id', i)}: {e}")
            continue
        
    # Convert list to numpy array
    embeddings_array = np.array(recipe_embeddings_list)
    
    # Create new dataframe with only valid recipes
    valid_recipes_df = pd.DataFrame(valid_recipes_list)
    valid_recipes_df = valid_recipes_df.reset_index(drop=True)
    
    print(f"Created {len(embeddings_array)} recipe embeddings")
    return embeddings_array, valid_recipes_df

def save_all_files(recipes_df, recipe_embeddings, recipe_stats):
    print("Saving all files...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    np.save(f'recipe_embeddings_{timestamp}.npy', recipe_embeddings)
    print(f"Saved embeddings")
    
    # Save filtered recipes dataframe
    with open(f'filtered_recipes_{timestamp}.pkl', 'wb') as f:
        pickle.dump(recipes_df, f)
    print(f"Saved recipes.")
    
    # Save recipe statistics
    with open(f'recipe_statistics_{timestamp}.pkl', 'wb') as f:
        pickle.dump(recipe_stats, f)
    print(f"Saved statistics")
    
    print("All files saved successfully!")

def create_all_necessary_files(recipes_path, interactions_path, model_path):
    print("Starting full preprocessing pipeline")
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Load the trained model
    model = BertModel.from_pretrained('bert-base-uncased')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Set up tag categories and ingredient groups
    tag_categories = setup_tag_categories()
    ingredient_groups = setup_ingredient_groups()
    
    # Load and clean recipes
    recipes_df = load_and_clean_recipes(recipes_path)
    
    # Create recipe statistics
    recipe_stats = create_recipe_statistics(interactions_path)
    
    # Create recipe embeddings
    recipe_embeddings, filtered_recipes_df = create_recipe_embeddings(
        recipes_df, model, tokenizer, device, tag_categories, ingredient_groups
    )
    
    # Save all files
    save_all_files(filtered_recipes_df, recipe_embeddings, recipe_stats)

if __name__ == "__main__":
    create_all_necessary_files(
        recipes_path='RAW_recipes.csv',
        interactions_path='RAW_interactions.csv',
        model_path='tag_based_bert_model.pth'
    )
    
    print("All preprocessing complete! You can now use the search system.") 