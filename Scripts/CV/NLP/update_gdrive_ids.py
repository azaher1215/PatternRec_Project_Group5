#!/usr/bin/env python3
"""
Update Google Drive File IDs in Streamlit App
=============================================

This script helps you update the Google Drive file IDs in your Streamlit app.
"""

import re
import os

def extract_file_id(url):
    """Extract file ID from Google Drive URL"""
    # Pattern for Google Drive URLs
    patterns = [
        r'https://drive\.google\.com/file/d/([a-zA-Z0-9_-]+)',
        r'https://drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)',
        r'https://drive\.google\.com/uc\?id=([a-zA-Z0-9_-]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    # If it's already just a file ID
    if re.match(r'^[a-zA-Z0-9_-]+$', url):
        return url
    
    return None

def update_file_ids():
    """Update Google Drive file IDs in the Streamlit app"""
    
    app_file = 'streamlit_app_gdrive.py'
    
    if not os.path.exists(app_file):
        print(f"❌ {app_file} not found!")
        print("Please make sure you're in the correct directory.")
        return
    
    print("🔗 Google Drive File ID Updater")
    print("=" * 35)
    print()
    
    # File mapping
    files_to_update = {
        'tag_based_bert_model.pth': 'Model file (.pth)',
        'RAW_recipes.csv': 'Recipes CSV file',
        'recipe_statistics_231630.pkl': 'Statistics file',
        'recipe_scores_231630.pkl': 'Scores file'
    }
    
    # Get file IDs from user
    new_ids = {}
    
    for filename, description in files_to_update.items():
        print(f"📄 {description}")
        print(f"   File: {filename}")
        
        while True:
            url_or_id = input("   Enter Google Drive URL or file ID: ").strip()
            
            if not url_or_id:
                print("   ⚠️  Skipping this file...")
                break
            
            file_id = extract_file_id(url_or_id)
            
            if file_id:
                new_ids[filename] = file_id
                print(f"   ✅ File ID: {file_id}")
                break
            else:
                print("   ❌ Invalid URL or file ID. Please try again.")
        
        print()
    
    if not new_ids:
        print("❌ No file IDs provided. Exiting.")
        return
    
    # Read the current file
    with open(app_file, 'r') as f:
        content = f.read()
    
    # Update the file IDs
    updated_content = content
    
    for filename, file_id in new_ids.items():
        # Pattern to find the line with this filename
        pattern = rf"('{filename}':\s*')[^']*(')"
        replacement = rf"\g<1>{file_id}\g<2>"
        updated_content = re.sub(pattern, replacement, updated_content)
        print(f"✅ Updated {filename} → {file_id}")
    
    # Write the updated content
    with open(app_file, 'w') as f:
        f.write(updated_content)
    
    print()
    print("🎉 File IDs updated successfully!")
    print(f"📄 Updated {app_file}")
    print()
    print("🔗 Next steps:")
    print("   1. Test locally: streamlit run streamlit_app_gdrive.py")
    print("   2. Commit to GitHub: git add . && git commit -m 'Update Google Drive IDs'")
    print("   3. Deploy to Streamlit Cloud!")

def main():
    """Main function"""
    print("🚀 Google Drive File ID Updater")
    print("=" * 32)
    print()
    print("This script will help you update Google Drive file IDs in your Streamlit app.")
    print()
    
    # Check if embeddings file ID is already set
    embeddings_id = '1PSidY1toSfgECXDxa4pGza56Jq6vOq6t'
    print(f"✅ torch_recipe_embeddings_231630.pt: {embeddings_id} (already set)")
    print()
    
    print("Please provide Google Drive URLs or file IDs for the remaining files:")
    print("(Press Enter to skip a file)")
    print()
    
    update_file_ids()

if __name__ == "__main__":
    main() 