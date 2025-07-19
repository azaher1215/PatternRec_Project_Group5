import pandas as pd
from ast import literal_eval
from transformers import BertTokenizer, BertModel
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch
import os
from sklearn.model_selection import train_test_split
import random
import re

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

def create_pair_data(recipes_df: pd.DataFrame, interactions_df: pd.DataFrame ,num_pairs: int = 15000):
    # This function creates the training pairs for the model. 
    # we first analyzed the data to create catogeries for the tags and ingredients. Under each of these, we have a list for cuisine, dietery, poultry, etc.
    # As we trained the model, we found that the model was not able to learn the tags and ingredients so we created a structured text represenation so it can easily learn. 
    # the prompt used is: Analyze the two csv files attached and created a structured text representation to be used for training a bert model to understand
    # tags and ingredients such that if a user later searches for a quick recipe, it can be used to find a recipe that is quick to make. 
  
  # Set up the structured text categories and groups
  tag_categories = setup_tag_categories()
  ingredient_groups = setup_ingredient_groups()
  
  # Make a list to store all our pairs
  pair_data_list = []
  
  # create the pairs
  for pair_number in range(num_pairs):
    
    #Pick a random recipe from our dataframe
    random_recipe_data = recipes_df.iloc[random.randint(0, len(recipes_df) - 1)]
    
    # Get the tags from this recipe
    recipe_tags_list = random_recipe_data['tags']
    
    # Select some random tags (maximum 5, but maybe less if recipe has fewer tags)
    num_tags_to_select = min(5, len(recipe_tags_list))
    selected_tags_list = []
    
    # Pick random sample of tags and join them to a query string
    selected_tags_list = random.sample(recipe_tags_list, num_tags_to_select)
    
    # Create the positive recipe text using structured format
    positive_recipe_text = create_structured_recipe_text(random_recipe_data, tag_categories, ingredient_groups)
    
    # Find a negative recipe that has less than 2 tags in common with the query
    anchor = ' '.join(selected_tags_list)
    anchor_tags_set = set(anchor.split())
    
    negative_recipe_text = None
    attempts_counter = 0
    max_attempts_allowed = 100
    
    # Keep trying until we find a good negative recipe (Added a max attempts to avoid infinite loop)
    while negative_recipe_text is None and attempts_counter < max_attempts_allowed:
      random_negative_recipe = recipes_df.iloc[random.randint(0, len(recipes_df) - 1)]
      
      # Get tags from this negative recipe
      negative_recipe_tags = random_negative_recipe['tags']
      negative_recipe_tags_set = set(negative_recipe_tags)
      
      # Count how many tags overlap
      overlap_count = 0
      for anchor_tag in anchor_tags_set:
        if anchor_tag in negative_recipe_tags_set:
          overlap_count = overlap_count + 1
      
      attempts_counter = attempts_counter + 1
      
      # If overlap is small enough (2 or less), we can use this as negative
      if overlap_count <= 2:
        # Create the negative recipe text using structured format
        negative_recipe_text = create_structured_recipe_text(random_negative_recipe, tag_categories, ingredient_groups)
        
        print(f"Found all negative recipes. Overlap: {overlap_count}")
        break

    # If we found a negative recipe, add this pair to our list
    if negative_recipe_text is not None:
      # Create a tuple with the three parts
      pair_data_list.append((anchor, positive_recipe_text, negative_recipe_text))
      print(f"Created pair {pair_number + 1}: Anchor='{anchor}', Overlap={overlap_count}")
    else:
      print(f"Could not find negative recipe for anchor '{anchor}' after {max_attempts_allowed} attempts")

    # Show progress every 1000 pairs
    if (pair_number + 1) % 1000 == 0:
      print(f"Progress: Created {pair_number + 1}/{num_pairs} pairs")

  # Convert our list to a pandas DataFrame and return it
  result_dataframe = pd.DataFrame(pair_data_list, columns=['anchor', 'positive', 'negative'])
  
  print(f"Final result: Created {len(result_dataframe)} pairs total")
  return result_dataframe

class pos_neg_pair_dataset(Dataset):
  #typical dataset class to tokenize for bert model and return the ids and masks
  def __init__(self, pair_data, tokenizer, max_length=128):
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
    #evaluation method, same as training but with no gradient updates
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

def train_model(train_loader, num_epochs=3):
    # initialize the model, criterion, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.to(device)
    criterion = nn.TripletMarginLoss(margin=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            #load the ids and masks to device 
            anchor_input_ids = batch['anchor_input_ids'].to(device)
            anchor_attention_mask = batch['anchor_attention_mask'].to(device)
            positive_input_ids = batch['positive_input_ids'].to(device)
            positive_attention_mask = batch['positive_attention_mask'].to(device)
            negative_input_ids = batch['negative_input_ids'].to(device)
            negative_attention_mask = batch['negative_attention_mask'].to(device)

            # get the embeddings to extract the [CLS] token embeddings
            model(anchor_input_ids,anchor_attention_mask)
            anchor_outputs = model(anchor_input_ids, anchor_attention_mask)
            positive_outputs = model(positive_input_ids, positive_attention_mask)
            negative_outputs = model(negative_input_ids, negative_attention_mask)

            # Extract the[CLS] token embeddings
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

        # per batch average loss total loss / number of batches
        print(f'Epoch {epoch+1}, Average Loss: {total_loss/len(train_loader):.4f}')

    return model

if __name__ == '__main__':

  if not os.path.exists('pair_data.parquet'):
    # Load and prepare the data
    print("Loading recipe data")
    recipes_df = pd.read_csv('RAW_recipes.csv')

    # Clean the data
    recipes_df['name'] = recipes_df['name'].apply(clean_text)
    recipes_df['tags'] = recipes_df['tags'].apply(literal_eval)
    recipes_df['ingredients'] = recipes_df['ingredients'].apply(literal_eval)

    # Filter recipes with meaningful data (no empty tags)
    recipes_df = recipes_df[recipes_df['tags'].str.len() > 0]

    # Load interactions
    print("Loading interaction data")
    interactions_df = pd.read_csv('RAW_interactions.csv')
    interactions_df = interactions_df.dropna(subset=['rating'])
    interactions_df['rating'] = pd.to_numeric(interactions_df['rating'], errors='coerce')
    interactions_df = interactions_df.dropna(subset=['rating'])

    # Create training pairs
    pair_data = create_pair_data(recipes_df, interactions_df, num_pairs=15000)

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
  model = train_model(train_loader, num_epochs=3)

  #evaluate the model
  print("Evaluating model...")
  evaluate_model(model, val_loader)

  # Save model
  torch.save(model.state_dict(), 'tag_based_bert_model.pth')
  print("Model saved to tag_based_bert_model.pth")
  print("Training Complete")