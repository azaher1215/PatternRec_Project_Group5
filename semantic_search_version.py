import pandas as pd
import ast
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

def clean_text(text):
    # convert to lowercase and strip whitespace
    if not isinstance(text, str): #used to check if the text is a string
        return ''
    text = text.lower()
    return text.strip()

def parse_str_list(s):
    """
    Safely parse a stringified Python list (e.g., "['a', 'b']") into a Python list.
    Returns an empty list if parsing fails.
    """
    try:
        return [x.strip().lower() for x in ast.literal_eval(s)]
    except Exception:
        return []

def load_recipes(path): 
    # load recipes from csv
    converters = {
        'tags': parse_str_list,
        'ingredients': parse_str_list
    }
    df = pd.read_csv(path, converters=converters, usecols=[
        'id', 'name', 'tags', 'ingredients', 'description', 'minutes', 'n_ingredients'
    ])
    # Clean text fields
    df['name'] = df['name'].apply(clean_text)
    df['description'] = df['description'].apply(clean_text)
    # Remove duplicates and normalize tags/ingredients
    df['tags'] = df['tags'].apply(lambda tags: sorted(set(tags)))
    df['ingredients'] = df['ingredients'].apply(lambda ings: sorted(set(ings)))
    print(df['ingredients'].head())
    return df

def load_interactions(path):
    # load user interactions from csv
    df = pd.read_csv(path, usecols=['user_id', 'recipe_id', 'rating'])
    return df

def aggregate_ratings(interactions):
    # compute average rating and count per recipe
    agg = interactions.groupby('recipe_id')['rating'].agg(['mean', 'count']).reset_index()
    agg.rename(columns={'mean': 'avg_rating', 'count': 'rating_count'}, inplace=True)
    return agg

class semantic_search_dataset(Dataset):
    def __init__(self, recipes, tokenizer, max_length=128):
        self.recipes = recipes
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.recipes)
    
    def __getitem__(self, idx):
        recipe = self.recipes.iloc[idx]

        # Create comprehensive recipe text representation
        recipe_text = f"Recipe: {recipe['name']}. "
        if pd.notna(recipe['description']):
            recipe_text += f"Description: {recipe['description']}. "
        if recipe['ingredients']:
            recipe_text += f"Ingredients: {', '.join(recipe['ingredients'])}. "
        if recipe['tags']:
            recipe_text += f"Tags: {', '.join(recipe['tags'])}."
        
        # Create positive query from recipe tags
        positive_query = " ".join(recipe['tags'][:5])  # Use first 5 tags as positive query
        
        # Create negative query from different recipe
        neg_idx = np.random.choice([i for i in range(len(self)) if i != idx])
        neg_recipe = self.recipes.iloc[neg_idx]
        negative_query = " ".join(neg_recipe['tags'][:5])
        
        # Tokenize recipe text
        recipe_tokens = self.tokenizer(
            recipe_text, 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_length, 
            return_tensors='pt'
        )
        
        # Tokenize positive query
        pos_tokens = self.tokenizer(
            positive_query,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Tokenize negative query
        neg_tokens = self.tokenizer(
            negative_query,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'recipe': recipe_tokens,
            'positive_query': pos_tokens,
            'negative_query': neg_tokens,
            'recipe_id': recipe['id']
        }

class RecipeBERT(torch.nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased'):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.projection = torch.nn.Linear(768, 768)
        
    def encode_text(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        embeddings = self.projection(embeddings)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
    
    def forward(self, recipe_tokens, query_tokens):
        recipe_emb = self.encode_text(
            recipe_tokens['input_ids'].squeeze(),
            recipe_tokens['attention_mask'].squeeze()
        )
        query_emb = self.encode_text(
            query_tokens['input_ids'].squeeze(),
            query_tokens['attention_mask'].squeeze()
        )
        return recipe_emb, query_emb

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, recipe_emb, pos_query_emb, neg_query_emb):
        pos_sim = F.cosine_similarity(recipe_emb, pos_query_emb, dim=1)
        neg_sim = F.cosine_similarity(recipe_emb, neg_query_emb, dim=1)
        loss = torch.clamp(self.margin - pos_sim + neg_sim, min=0)
        return loss.mean()

if __name__ == '__main__':
    
    # region 1: loading data and aggregating ratings
    # File paths
    recipes_path = 'RAW_recipes.csv'
    interactions_path = 'RAW_interactions.csv'

    # Load data (recipes and interactions)
    recipes = load_recipes(recipes_path)
    print(f'Recipes loaded: {len(recipes)}')
    interactions = load_interactions(interactions_path)
    print(f'Interactions loaded: {len(interactions)}')

    # Aggregate ratings for each recipe (avg rating and count) to use later
    ratings = aggregate_ratings(interactions)
    print('finished aggregating ratings')

    # Merge ratings into recipes
    recipes = recipes.merge(ratings, left_on='id', right_on='recipe_id', how='left')

    # Save cleaned data for later use
    recipes.to_parquet('processed_recipes.parquet', index=False)
    print('merged recipes and ratings saved to processed_recipes.parquet')
    # endregion 1

    # region 2: Bert model training for semantic search
    # Bert initialization
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = RecipeBERT()

    # Data Split
    train_df, test_df = train_test_split(recipes, test_size=0.2, random_state=42)

    # Create datasets
    train_dataset = semantic_search_dataset(train_df, tokenizer)
    test_dataset = semantic_search_dataset(test_df, tokenizer)

    # Create dataloaders
    batch_size = 16
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    #train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    #initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = ContrastiveLoss()

    #training loop
    num_epochs = 3
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            recipe_tokens = {k: v.to(device) for k, v in batch['recipe'].items()}
            pos_tokens = {k: v.to(device) for k, v in batch['positive_query'].items()}
            neg_tokens = {k: v.to(device) for k, v in batch['negative_query'].items()}
            
            recipe_emb, pos_query_emb = model(recipe_tokens, pos_tokens)
            _, neg_query_emb = model(recipe_tokens, neg_tokens)
            
            loss = criterion(recipe_emb, pos_query_emb, neg_query_emb)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f'Epoch {epoch}, Loss: {loss.item()}')

        #evaluate
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in test_dataloader:
                recipe_tokens = {k: v.to(device) for k, v in batch['recipe'].items()}
                pos_tokens = {k: v.to(device) for k, v in batch['positive_query'].items()}
                neg_tokens = {k: v.to(device) for k, v in batch['negative_query'].items()}
                
                recipe_emb, pos_query_emb = model(recipe_tokens, pos_tokens)
                _, neg_query_emb = model(recipe_tokens, neg_tokens)
                
                loss = criterion(recipe_emb, pos_query_emb, neg_query_emb)
                total_val_loss += loss.item()
                print(f'Epoch {epoch}, Validation Loss: {loss.item()}')

        avg_train_loss = total_loss / len(train_dataloader)
        avg_val_loss = total_val_loss / len(test_dataloader)
        print(f'Epoch {epoch}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_semantic_model.pt')
            print('New best model saved!')

    #save model
    torch.save(model.state_dict(), 'semantic_search_model.pth')
    print('Semantic search model saved to semantic_search_model.pth')
    # endregion 2