import streamlit as st
from PIL import Image
from model.classifier import get_model, predict, get_model_by_name
from model.search_script import search_for_recipes
import streamlit.components.v1 as components
import base64
import config as config
from utils.layout import render_layout

MODEL_PATH_MAP = {
    "Onion": config.MODEL_PATH_ONION,
    "Pear": config.MODEL_PATH_PEAR,
    "Strawberry": config.MODEL_PATH_STRAWBERRY,
    "Tomato": config.MODEL_PATH_TOMATO
}

VARIATION_CLASS_MAP = {
    "Onion": ['halved', 'sliced', 'whole'],
    "Strawberry": ['Hulled', 'sliced', 'whole'],
    "Tomato": ['diced', 'vines', 'whole'],
    "Pear": ['halved', 'sliced', 'whole']
}

@st.cache_resource
def load_model():
    return get_model()

@st.cache_resource
def load_model_variation(product_name):
    model_path = MODEL_PATH_MAP[product_name]
    num_classes = len(VARIATION_CLASS_MAP[product_name])
    return get_model_by_name(model_path, num_classes=num_classes)

def classification_and_recommendation_page():
    st.markdown("## 🍽️ Recipe Recommendation System")
    st.markdown("""
    <div class="about-box">
    <b>Recipe Recommendation Guide</b><br><br>

    Upload one or more food images. This module classifies each image into 
    <b>Onion, Pear, Strawberry, or Tomato</b> using <b>EfficientNet-B0</b>, and recommends recipes
    based on the combined classification results, using a fine-tuned BERT model.<br><br>

    <b>Steps:</b><br>
    1️⃣ Upload images (single or multiple) of produce, or directly add tags for recipe search.<br>
    2️⃣ Once uploaded, the corresponding produce tag will be automatically added to the search.<br>
    3️⃣ Use the sliders to choose the number of results and minimum recipe rating.<br>
    4️⃣ Click <b>"Search Recipe"</b> to view personalized recommendations.
    </div></br>                       
    """, unsafe_allow_html=True)



    model = load_model()

    uploaded_files = st.file_uploader("📤 Upload images (JPG/PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if "uploaded_images" not in st.session_state:
        st.session_state.uploaded_images = []
    if "image_tags" not in st.session_state:
        st.session_state.image_tags = {}
    if "image_variations" not in st.session_state:
        st.session_state.image_variations = {}

    if uploaded_files:
        for img_file in uploaded_files:
            if img_file.name not in [img.name for img in st.session_state.uploaded_images]:
                img = Image.open(img_file).convert("RGB")
                label, main_class_prob = predict(img, model)

                variation = None
                if label in VARIATION_CLASS_MAP:
                    variation_model = load_model_variation(label)
                    class_labels = VARIATION_CLASS_MAP[label]
                    variation_label, var_conf = predict(img, variation_model, class_labels=class_labels)
                    variation = f"{variation_label} ({var_conf*main_class_prob* 100:.1f}%)"

                st.session_state.uploaded_images.append(img_file)
                st.session_state.image_tags[img_file.name] = label
                st.session_state.image_variations[img_file.name] = variation

    current_file_names = [f.name for f in uploaded_files] if uploaded_files else []
    st.session_state.uploaded_images = [f for f in st.session_state.uploaded_images if f.name in current_file_names]
    st.session_state.image_tags = {k: v for k, v in st.session_state.image_tags.items() if k in current_file_names}
    st.session_state.image_variations = {k: v for k, v in st.session_state.image_variations.items() if k in current_file_names}
    
    if st.session_state.uploaded_images:
        html = """
        <style>
        .image-grid { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 10px; }
        .image-card {
            width: 140px; height: 200px;
            border: 1px solid #ccc; border-radius: 10px;
            overflow: hidden; text-align: center;
            font-size: 13px; position: relative;
            background: #fdfdfd; box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        }
        .image-card img {
            max-width: 100%; max-height: 110px;
            object-fit: contain; margin-top: 5px;
        }
        </style>
        <div class="image-grid">
        """

        for img in st.session_state.uploaded_images:
            label = st.session_state.image_tags.get(img.name, "unknown")
            variation = st.session_state.image_variations.get(img.name, "")
            combined_label = f"{label.upper()} </br> {variation}" if variation else label.upper()
            img_b64 = base64.b64encode(img.getvalue()).decode()

            html += f"""
            <div class="image-card">
                <img src="data:image/png;base64,{img_b64}" />
                <div style="margin-top: 5px; font-weight: bold; font-size: 13px;">{combined_label}</div>
                <div style="color:gray; font-size:11px;">{img.name}</div>
            </div>
            """

        html += "</div>"
        grid_rows = ((len(st.session_state.uploaded_images) - 1) // 5 + 1)
        components.html(html, height=200 * grid_rows + 40, scrolling=True)

    st.markdown("---")
    st.markdown("## 🔍 Recipe Recommendation")

    if 'search_system' not in st.session_state:
        with st.spinner("Initializing recipe search system"):
            st.session_state.search_system = search_for_recipes()

    search_system = st.session_state.search_system

    if not search_system.is_ready:
        st.error("System not ready. Please check data files and try again.")
        return

    unique_tags = list(set(st.session_state.image_tags.values()))
    default_query = " ".join(unique_tags)

    query = st.text_input(
        "Search for recipes:",
        value=default_query,
        placeholder="e.g., 'onion tomato pasta', 'strawberry dessert', etc."
    )

    col1, col2 = st.columns(2)
    with col1:
        num_results = st.slider("Number of results", 1, 15, 5)
    with col2:
        min_rating = st.slider("Minimum rating", 1.0, 5.0, 3.0, 0.1)

    if st.button("🔍 Search Recipes") and query:
        with st.spinner(f"Searching for '{query}'..."):
            results = search_system.search_recipes(query, num_results, min_rating)

        if results:
            st.markdown(f"### Top {len(results)} recipe recommendations for: *'{query}'*")
            st.markdown("<hr>", unsafe_allow_html=True)

            for i, recipe in enumerate(results, 1):
                steps_html = "".join([f"<li>{step.strip().capitalize()}</li>" for step in recipe.get("steps", [])])
                description = recipe.get("description", "").strip().capitalize()

                html_code = f"""
                <div style="margin: 8px 0; padding: 8px; border-radius: 12px; background-color: #fdfdfd; 
                            box-shadow: 0 2px 8px rgba(0,0,0,0.06); font-family: Arial, sans-serif; 
                            border: 1px solid #e0e0e0;">
                    <div style="font-size: 18px; font-weight: bold; color: #333; margin-bottom: 8px;">
                        {i}. {recipe['name']}
                    </div>
                    <div style="margin: 4px 0 12px 0; font-size: 14px; color: #555;">
                        <b>{recipe['minutes']} min</b> &nbsp;&nbsp;|&nbsp;&nbsp;
                        <b>{recipe['n_steps']} steps</b> &nbsp;&nbsp;|&nbsp;&nbsp;
                        <b>{recipe['avg_rating']:.1f}/5.0</b>
                        <span style="font-size: 12px; color: #999;">({recipe['num_ratings']} ratings)</span>
                    </div>
                    <div style="margin-bottom: 8px; font-size: 14px;">
                        <b>Match Score:</b> 
                        <span style="color: #007acc; font-weight: bold;">{recipe['similarity_score']:.1%}</span>
                        <span style="font-size: 12px; color: #888;">(query match)</span>
                    </div>
                    <div style="margin-bottom: 8px;">
                        <b>Tags:</b><br>
                        <div style="margin-top: 8px;">
                            {" ".join([f"<span style='background:#eee;padding:4px 8px;border-radius:6px;margin:2px;display:inline-block;font-size:12px'>{tag}</span>" for tag in recipe['tags']])}
                        </div>
                    </div>
                    <div style="margin-bottom: 8px;">
                        <b>Ingredients:</b><br>
                        <span style="font-size: 13px; color: #444; display: block;">
                            {', '.join(recipe['ingredients'][:8])}{'...' if len(recipe['ingredients']) > 8 else ''}
                        </span>
                    </div>
                    {f"<div style='margin-top: 10px; font-size: 13px; color: #333;'><b>Description:</b><br><span>{description}</span></div>" if description else ""}
                    {f"<div style='margin-top: 10px; font-size: 13px;'><b>Steps:</b><ol style='margin: 6px 0 0 18px; padding: 0;'>{steps_html}</ol></div>" if steps_html else ""}
                </div>
                """
                components.html(html_code, height=340, scrolling=True)
        else:
            st.warning(f"No recipes found for '{query}' with a minimum rating of {min_rating}/5.0.")

render_layout(classification_and_recommendation_page)
