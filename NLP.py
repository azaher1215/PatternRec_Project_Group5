import pandas as pd
import ast



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
