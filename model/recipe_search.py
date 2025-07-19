import os
import csv
import ast
import pickle
import gdown
import torch
import torch.nn.functional as F
import streamlit as st
from transformers import BertTokenizer, BertModel
from config import GOOGLE_DRIVE_FILES


def download_file_from_drive(file_id: str, destination: str, file_name: str) -> bool:
    try:
        with st.spinner(f"Downloading {file_name}..."):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, destination, quiet=False)
        return True
    except Exception as e:
        st.error(f"Failed to download {file_name}: {e}")
        return False

def ensure_files_downloaded():
    for filename, file_id in GOOGLE_DRIVE_FILES.items():
        if not os.path.exists(filename):
            success = download_file_from_drive(file_id, filename, filename)
            if not success:
                return False
    return True

class GoogleDriveRecipeSearch:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not ensure_files_downloaded():
            self.is_ready = False
            return

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased")

        if os.path.exists("assets/nlp/tag_based_bert_model.pth"):
            self.model.load_state_dict(
                torch.load("assets/nlp/tag_based_bert_model.pth", map_location=self.device)
            )
            st.success("Trained model loaded successfully!")
        else:
            st.warning("Using untrained model")

        self.model.to(self.device)
        self.model.eval()

        self.load_data()
        self.is_ready = True

    def load_data(self):
        self.recipe_embeddings = torch.load("assets/nlp/torch_recipe_embeddings_231630.pt", map_location=self.device)
        self.recipes = self._load_recipes("assets/nlp/RAW_recipes.csv")
        self.recipe_stats = pickle.load(open("assets/nlp/recipe_statistics_231630.pkl", "rb"))
        self.recipe_scores = pickle.load(open("assets/nlp/recipe_scores_231630.pkl", "rb"))

    def _load_recipes(self, path):
        recipes = []
        with open(path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for idx, row in enumerate(reader):
                name = row.get("name", "").strip()
                if not name or name.lower() in ["nan", "unknown recipe"]:
                    continue
                try:
                    recipe = {
                        "id": int(row.get("id", idx)),
                        "name": name,
                        "ingredients": ast.literal_eval(row.get("ingredients", "[]")),
                        "tags": ast.literal_eval(row.get("tags", "[]")),
                        "minutes": int(float(row.get("minutes", 0))),
                        "n_steps": int(float(row.get("n_steps", 0))),
                        "description": row.get("description", ""),
                        "steps": ast.literal_eval(row.get("steps", "[]"))
                    }
                    recipes.append(recipe)
                except:
                    continue
        return recipes

    def search_recipes(self, query, num_results=5, min_rating=3.0):
        if not query.strip():
            return []
        print('im here')

        tokens = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True)
        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        with torch.no_grad():
            outputs = self.model(**tokens)
            query_embedding = outputs.last_hidden_state[:, 0, :]

        query_embedding = F.normalize(query_embedding, dim=1)
        recipe_embeddings = F.normalize(self.recipe_embeddings, dim=1)

        similarity_scores = torch.matmul(recipe_embeddings, query_embedding.T).squeeze()

        final_scores = []
        for i in range(len(self.recipe_embeddings)):
            recipe = self.recipes[i]
            avg_rating, num_ratings, *_ = self.recipe_stats.get(recipe["id"], (0.0, 0, 0))
            if avg_rating < min_rating or num_ratings < 2:
                continue
            combined_score = (
                0.6 * similarity_scores[i].item() +
                0.4 * self.recipe_scores.get(recipe["id"], 0)
            )
            final_scores.append((combined_score, i))

        top_matches = sorted(final_scores, key=lambda x: x[0], reverse=True)[:num_results]

        results = []
        for score, idx in top_matches:
            recipe = self.recipes[idx]
            avg_rating, num_ratings, *_ = self.recipe_stats.get(recipe["id"], (0.0, 0, 0))
            results.append({
                "name": recipe["name"],
                "tags": recipe.get("tags", []),
                "ingredients": recipe.get("ingredients", []),
                "minutes": recipe.get("minutes", 0),
                "n_steps": recipe.get("n_steps", 0),
                "avg_rating": avg_rating,
                "num_ratings": num_ratings,
                "similarity_score": similarity_scores[idx].item(),
                "combined_score": score,
                "steps": recipe.get("steps", []),
                "description": recipe.get("description", "")
            })

        return results

@st.cache_resource
def load_search_system():
    return GoogleDriveRecipeSearch()
