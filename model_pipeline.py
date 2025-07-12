import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, AdamW
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
from typing import List, Tuple, Dict, Optional

# --- Data Preparation ---
class RecipeDataset(Dataset):
    """
    Dataset for recipe-query pairs with contrastive learning.
    Creates positive pairs (recipe + relevant query) and negative pairs (recipe + irrelevant query).
    """
    
    def __init__(self, recipes_df: pd.DataFrame, tokenizer, max_length: int = 128):
        self.recipes_df = recipes_df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.recipe_ids = recipes_df['id'].tolist()
        
        # Create recipe text representations
        self.recipe_texts = []
        for _, row in recipes_df.iterrows():
            # Combine name, description, ingredients, and tags
            recipe_text = f"Recipe: {row['name']}. "
            if pd.notna(row['description']):
                recipe_text += f"Description: {row['description']}. "
            if row['ingredients']:
                recipe_text += f"Ingredients: {', '.join(row['ingredients'])}. "
            if row['tags']:
                recipe_text += f"Tags: {', '.join(row['tags'])}."
            
            self.recipe_texts.append(recipe_text)
    
    def __len__(self):
        return len(self.recipe_ids)
    
    def __getitem__(self, idx):
        recipe_text = self.recipe_texts[idx]
        
        # Create positive query from recipe tags/ingredients
        recipe_row = self.recipes_df.iloc[idx]
        positive_tags = recipe_row['tags'][:5]  # Take first 5 tags as positive query
        positive_query = " ".join(positive_tags)
        
        # Create negative query from different recipe
        neg_idx = np.random.choice([i for i in range(len(self)) if i != idx])
        neg_row = self.recipes_df.iloc[neg_idx]
        negative_tags = neg_row['tags'][:5]
        negative_query = " ".join(negative_tags)
        
        # Tokenize
        recipe_tokens = self.tokenizer(
            recipe_text, 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_length, 
            return_tensors='pt'
        )
        
        positive_tokens = self.tokenizer(
            positive_query,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        negative_tokens = self.tokenizer(
            negative_query,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'recipe': recipe_tokens,
            'positive_query': positive_tokens,
            'negative_query': negative_tokens,
            'recipe_id': self.recipe_ids[idx]
        }

# --- Model Architecture ---
class RecipeBERT(nn.Module):
    """
    BERT-based model for recipe-query semantic matching.
    Uses a shared BERT encoder for both recipes and queries.
    """
    
    def __init__(self, bert_model_name: str = 'bert-base-uncased', embedding_dim: int = 768):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.embedding_dim = embedding_dim
        
        # Projection layer to normalize embeddings
        self.projection = nn.Linear(embedding_dim, embedding_dim)
        
    def encode_text(self, input_ids, attention_mask):
        """Encode text using BERT and return normalized embeddings."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        embeddings = outputs.last_hidden_state[:, 0, :]
        # Project and normalize
        embeddings = self.projection(embeddings)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
    
    def forward(self, recipe_tokens, query_tokens):
        """Forward pass for recipe-query pairs."""
        recipe_embeddings = self.encode_text(
            recipe_tokens['input_ids'].squeeze(),
            recipe_tokens['attention_mask'].squeeze()
        )
        query_embeddings = self.encode_text(
            query_tokens['input_ids'].squeeze(),
            query_tokens['attention_mask'].squeeze()
        )
        return recipe_embeddings, query_embeddings

# --- Contrastive Loss ---
class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for training recipe-query embeddings.
    Minimizes distance between positive pairs, maximizes distance between negative pairs.
    """
    
    def __init__(self, temperature: float = 0.1, margin: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(self, recipe_emb, pos_query_emb, neg_query_emb):
        # Positive pair similarity
        pos_sim = F.cosine_similarity(recipe_emb, pos_query_emb, dim=1)
        
        # Negative pair similarity
        neg_sim = F.cosine_similarity(recipe_emb, neg_query_emb, dim=1)
        
        # Contrastive loss: maximize positive similarity, minimize negative similarity
        loss = torch.clamp(self.margin - pos_sim + neg_sim, min=0)
        return loss.mean()

# --- Training Functions ---
def train_model(model, train_loader, val_loader, device, num_epochs=3, lr=2e-5):
    """
    Train the RecipeBERT model with contrastive learning.
    """
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = ContrastiveLoss()
    
    model.to(device)
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_steps = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            recipe_tokens = {k: v.to(device) for k, v in batch['recipe'].items()}
            pos_tokens = {k: v.to(device) for k, v in batch['positive_query'].items()}
            neg_tokens = {k: v.to(device) for k, v in batch['negative_query'].items()}
            
            recipe_emb, pos_query_emb = model(recipe_tokens, pos_tokens)
            _, neg_query_emb = model(recipe_tokens, neg_tokens)
            
            loss = criterion(recipe_emb, pos_query_emb, neg_query_emb)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
        
        # Validation
        model.eval()
        val_loss = 0
        val_steps = 0
        
        with torch.no_grad():
            for batch in val_loader:
                recipe_tokens = {k: v.to(device) for k, v in batch['recipe'].items()}
                pos_tokens = {k: v.to(device) for k, v in batch['positive_query'].items()}
                neg_tokens = {k: v.to(device) for k, v in batch['negative_query'].items()}
                
                recipe_emb, pos_query_emb = model(recipe_tokens, pos_tokens)
                _, neg_query_emb = model(recipe_tokens, neg_tokens)
                
                loss = criterion(recipe_emb, pos_query_emb, neg_query_emb)
                val_loss += loss.item()
                val_steps += 1
        
        avg_train_loss = train_loss / train_steps
        avg_val_loss = val_loss / val_steps
        
        print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_recipe_bert.pt')
            print(f'New best model saved!')

# --- Main Training Pipeline ---
def main():
    # Load cleaned data
    print("Loading cleaned recipes...")
    recipes_df = pd.read_parquet('cleaned_recipes.parquet')
    print(f"Loaded {len(recipes_df)} recipes")
    
    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = RecipeBERT()
    
    # Create dataset
    dataset = RecipeDataset(recipes_df, tokenizer)
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_model(model, train_loader, val_loader, device, num_epochs=3)
    
    print("Training complete! Model saved as 'best_recipe_bert.pt'")

if __name__ == '__main__':
    main() 