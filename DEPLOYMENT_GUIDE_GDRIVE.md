# 🚀 Google Drive Streamlit App Deployment Guide

## ⚠️ Important Note about GitHub Pages

**GitHub Pages (username.github.io) CANNOT host Streamlit apps** because:
- GitHub Pages only hosts **static websites** (HTML, CSS, JavaScript)
- Streamlit apps require a **Python server** to run
- GitHub Pages doesn't support Python execution

## 🌐 Proper Deployment Options

### Option 1: Streamlit Cloud (Recommended - FREE)

**✅ Best choice for Streamlit apps with Google Drive files**

1. **Prepare your app:**
   - Update Google Drive file IDs in `streamlit_app_gdrive.py`
   - Push to your GitHub repository

2. **Deploy to Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository: `PatternRec_Project_Group7`
   - Main file: `streamlit_app_gdrive.py`
   - Requirements: `requirements_gdrive.txt`
   - Click "Deploy"

3. **Your app will be live at:**
   ```
   https://YOUR_USERNAME-patternrec-project-group7-streamlit-app-gdrive-xyz.streamlit.app
   ```

### Option 2: Hugging Face Spaces (FREE)

1. **Create a Space:**
   - Go to [huggingface.co/spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Name: `group5-recipe-recommendation`
   - SDK: **Streamlit**
   - Hardware: CPU Basic (free)

2. **Upload files:**
   - Upload `streamlit_app_gdrive.py` as `app.py`
   - Upload `requirements_gdrive.txt` as `requirements.txt`

3. **Your app will be live at:**
   ```
   https://huggingface.co/spaces/YOUR_USERNAME/group5-recipe-recommendation
   ```

### Option 3: Railway (FREE tier available)

1. **Connect GitHub:**
   - Go to [railway.app](https://railway.app)
   - Sign in with GitHub
   - Create new project from GitHub repo

2. **Configure deployment:**
   - Select your repository
   - Add environment variables if needed
   - Railway will auto-detect Python and install dependencies

### Option 4: Render (FREE tier available)

1. **Create web service:**
   - Go to [render.com](https://render.com)
   - Connect your GitHub repository
   - Choose "Web Service"

2. **Configure:**
   - Build Command: `pip install -r requirements_gdrive.txt`
   - Start Command: `streamlit run streamlit_app_gdrive.py --server.port $PORT`

## 🔧 Setup Instructions

### Step 1: Get Google Drive File IDs

For each of your files, you need the Google Drive file ID:

1. **Upload files to Google Drive**
2. **Right-click each file → "Get link"**
3. **Set to "Anyone with the link can view"**
4. **Copy the file ID from the URL:**
   ```
   https://drive.google.com/file/d/FILE_ID_HERE/view?usp=sharing
   ```

### Step 2: Update File IDs in Code

Edit `streamlit_app_gdrive.py` and update this section:

```python
GOOGLE_DRIVE_FILES = {
    'torch_recipe_embeddings_231630.pt': '1PSidY1toSfgECXDxa4pGza56Jq6vOq6t',  # ✅ Already set
    'tag_based_bert_model.pth': 'YOUR_ACTUAL_MODEL_FILE_ID',           # ⚠️ Update this
    'RAW_recipes.csv': 'YOUR_ACTUAL_RECIPES_FILE_ID',                  # ⚠️ Update this  
    'recipe_statistics_231630.pkl': 'YOUR_ACTUAL_STATS_FILE_ID',       # ⚠️ Update this
    'recipe_scores_231630.pkl': 'YOUR_ACTUAL_SCORES_FILE_ID'           # ⚠️ Update this
}
```

### Step 3: Test Locally (Optional)

```bash
# Install dependencies
pip install -r requirements_gdrive.txt

# Run locally
streamlit run streamlit_app_gdrive.py

# Test at http://localhost:8501
```

### Step 4: Deploy to Streamlit Cloud

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Add Google Drive Streamlit app"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your repo and `streamlit_app_gdrive.py`
   - Deploy!

## 📋 File Requirements

Make sure you have these files in your repo:

```
📁 Your Repository
├── streamlit_app_gdrive.py        # Main app with Google Drive loading
├── requirements_gdrive.txt        # Dependencies including gdown
├── DEPLOYMENT_GUIDE_GDRIVE.md     # This guide
└── ... other project files
```

## 🛠️ Troubleshooting

### Common Issues:

**1. "Failed to retrieve file url"**
- Make sure Google Drive files are set to "Anyone with the link can view"
- Check that file IDs are correct (no extra characters)

**2. "File not found" errors**
- Verify all file IDs are updated in the code
- Test file accessibility by visiting the Google Drive links

**3. "Out of memory" errors**
- Use Streamlit Cloud with more resources
- Or reduce dataset size for free tiers

**4. App crashes on startup**
- Check Streamlit Cloud logs for error details
- Ensure all dependencies are in requirements_gdrive.txt

## 🎯 Expected Performance

- **First load:** 2-5 minutes (downloading files from Google Drive)
- **Subsequent loads:** 30-60 seconds (files cached)
- **Search speed:** 1-2 seconds per query
- **Memory usage:** ~1.5GB (works on most free tiers)

## 📊 Comparison: Deployment Options

| Platform | Cost | Ease of Use | Performance | Custom Domain |
|----------|------|-------------|-------------|---------------|
| Streamlit Cloud | FREE | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Hugging Face | FREE | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| Railway | FREE tier | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Render | FREE tier | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |

## 🎉 Final Result

Your deployed app will:
- ✅ Load files automatically from Google Drive
- ✅ Handle 231,630 recipes with semantic search
- ✅ Avoid all numpy version conflicts
- ✅ Provide professional interface
- ✅ Work on mobile devices
- ✅ Have a public URL you can share

## 📞 Support

If you encounter issues:
1. Check the deployment platform's logs
2. Verify Google Drive file permissions
3. Test locally first
4. Check that all file IDs are correct

---

**Ready to deploy? Start with Streamlit Cloud - it's the easiest option for Streamlit apps!** 🚀 