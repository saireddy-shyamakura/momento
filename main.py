import torch
import clip
from PIL import Image
import os
import pickle
import numpy as np


# Config
STORE_PATH = "image_store.pkl"
device = "cpu"  # change to "cuda" if GPU available

# Load CLIP model
model, preprocess = clip.load("ViT-B/32", device=device)

# In-memory store
paths = []
features_list = [] 
features_matrix = None

# Load & Save
def load_store():
    global paths, features_list

    if os.path.exists(STORE_PATH):
        with open(STORE_PATH, "rb") as f:
            paths, features_list = pickle.load(f)

        build_index()
        print(f"Loaded {len(paths)} images")
    else:
        paths = []
        features_list = []
        build_index()
        print("No existing store found, starting fresh")


def save_store():
    with open(STORE_PATH, "wb") as f:
        pickle.dump((paths, features_list), f)


# Feature Extraction
def extract_image_features(image_path):
    image_path = os.path.abspath(image_path)
    with torch.inference_mode():
        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

        features = model.encode_image(image)
        features = features / features.norm(dim=-1, keepdim=True)

        # convert to numpy 
        return features.squeeze(0).cpu().numpy().astype(np.float32)

def extract_text_features(text):
    with torch.inference_mode():
        text = clip.tokenize([text]).to(device=device)

        features = model.encode_text(text=text)
        features = features / features.norm(dim=-1, keepdim=True)

        return features.squeeze(0).cpu().numpy().astype(np.float32)

# Add Images
def add_images(*input_paths):
    image_paths = []
    for path in input_paths:
        path = os.path.abspath(path)
        if os.path.isdir(path):
            for file in os.listdir(path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif')):
                    image_paths.append(os.path.join(path, file))
        elif os.path.isfile(path):
            image_paths.append(path)
        else:
            print(f"Invalid path: {path}")

    added_any = False
    for img_path in image_paths:
        img_path = os.path.abspath(img_path)    
        exists = img_path in paths
        if exists:
            print(f"Skipped (already exists): {img_path}")
            continue
        
        try:
            features = extract_image_features(img_path)
            paths.append(img_path)
            features_list.append(features)
            print(f"Added: {img_path}")
            added_any = True
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    if added_any:
        save_store()
        build_index()

def build_index():
    global features_matrix
    if len(features_list) > 0:
        features_matrix = np.vstack(features_list)
    else:
        features_matrix = None


# Cosine Similarity
def cosine_similarity(a, b):
    return np.dot(a, b)

# Search (Image → Image)
def image_search(query_image_path, top_k=3):

    query_features = extract_image_features(query_image_path)

    if features_matrix is None:
        return []

    scores = features_matrix @ query_features

    top_idx = np.argsort(scores)[::-1][:top_k]

    return [(scores[i], paths[i]) for i in top_idx]

# Search (Text → Image)
def text_search(text, top_k=3):
    query_features = extract_text_features(text)

    if features_matrix is None:
        return []

    scores = features_matrix @ query_features

    top_idx = np.argsort(scores)[::-1][:top_k]

    return [(scores[i], paths[i]) for i in top_idx]


# Run
if __name__ == "__main__":
    load_store()

    # add images path as list or give a single folder
    # add_images("images/dog.jpg", "images/car.jpg", "images/person_running.jpg")
    add_images("images")

    # Search via image
    results = image_search("images/dog2.jpg", top_k=3)

    print("\nTop Matches (Image Search) :")
    for score, path in results:
        print(f"{path} -> {score:.4f}")

    # Search via text
    results = text_search("a person running",top_k = 3)

    print("\nTop Matches (Text Search):")
    for score, path in results:
        print(f"{path} -> {score:.4f}")