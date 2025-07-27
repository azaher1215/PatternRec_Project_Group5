from utils.layout import render_layout
import streamlit as st
import time
from model.search_script import search_for_recipes
import streamlit.components.v1 as components

def recipe_search_page():
    st.markdown("""
        ## Advanced Recipe Recommendation
        <div class="about-box">
        This module uses a custom-trained BERT model to semantically search recipes
        based on your query of ingredients and tags.
        </div>
    """, unsafe_allow_html=True)

    if 'search_system' not in st.session_state:
        with st.spinner("Initializing recipe search system"):
            st.session_state.search_system = search_for_recipes()

    search_system = st.session_state.search_system

    if not search_system.is_ready:
        st.error("System not ready. Please check data files and try again.")
        return

    query = st.text_input(
        "Search for recipes:",
        placeholder="e.g., 'chicken pasta italian', 'vegetarian salad', 'chocolate dessert', 'quick easy' "
    )

    col1, col2 = st.columns(2)
    with col1:
        num_results = st.slider("Number of results", 1, 15, 5)
    with col2:
        min_rating = st.slider("Minimum rating", 1.0, 5.0, 3.0, 0.1)

    if st.button("🔍 Search Recipes") and query:
        with st.spinner(f"Searching for '{query}'..."):
            start = time.time()
            print(query, num_results, min_rating)
            results = search_system.search_recipes(query, num_results, min_rating)
            elapsed = time.time() - start

        if results:
            st.markdown(f"### Top {len(results)} recipe recommendations for: *'{query}'*")
            st.markdown("<hr>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            columns = [col1, col2]
    
            for i, recipe in enumerate(results, 1):
                steps_html = "".join([f"<li>{step.strip().capitalize()}</li>" for step in recipe.get("steps", [])])
                description = recipe.get("description", "").strip().capitalize()

                html_code = f"""
                <div style=\"width: auto; margin: 8px 0 8px 0; padding: 8px; border-radius: 12px; background-color: #fdfdfd; box-shadow: 0 2px 8px rgba(0,0,0,0.06); font-family: Arial, sans-serif; border: 1px solid #e0e0e0;\">
                    <div style=\"font-size: 18px; font-weight: bold; color: #333; margin-bottom: 8px;\"> {i}. {recipe['name']}</div>

                    <div style=\"margin: 4px 0 12px 0; font-size: 14px; color: #555;\">
                         <b>{recipe['minutes']} min</b> &nbsp;&nbsp;|&nbsp;&nbsp; <b>{recipe['n_steps']} steps</b> &nbsp;&nbsp;|&nbsp;&nbsp; <b>{recipe['avg_rating']:.1f}/5.0</b>
                        <span style=\"font-size: 12px; color: #999;\">({recipe['num_ratings']} ratings)</span>
                    </div>

                    <div style=\"margin-bottom: 8px; font-size: 14px;\">
                        <b>Match Score:</b> <span style=\"color: #007acc; font-weight: bold;\">{recipe['similarity_score']:.1%}</span>
                        <span style=\"font-size: 12px; color: #888;\">(query match)</span>
                    </div>

                    <div style=\"margin-bottom: 8px;\">
                        <b>Tags:</b><br>
                        <div style=\"margin-top: 8px;\">
                            {" ".join([f"<span style='background:#eee;padding:4px 8px;border-radius:6px;margin:2px;display:inline-block;font-size:12px'>{tag}</span>" for tag in recipe['tags']])}
                        </div>
                    </div>

                    <div style=\"margin-bottom: 8px;\">
                        <b>Ingredients:</b><br>
                        <span style=\"font-size: 13px; color: #444; margin-top: 4px; display: block;\">{', '.join(recipe['ingredients'][:8])}
                        {'...' if len(recipe['ingredients']) > 8 else ''}</span>
                    </div>

                    {"<div style='margin-top: 10px; font-size: 13px; color: #333;'><b>Description:</b><br><span style='margin-top: 4px; display: block;'>" + description + "</span></div>" if description else ""}

                    {"<div style='margin-top: 10px; font-size: 13px;'><b>Steps:</b><ol style='margin: 6px 0 0 18px; padding: 0;'>" + steps_html + "</ol></div>" if steps_html else ""}
                </div>
                """
                with columns[(i-1) % 2]:
                    components.html(html_code, height=340, scrolling=True)

        else:
            st.warning(f"No recipes found for '{query}' with a minimum rating of {min_rating}/5.0.")

render_layout(recipe_search_page)