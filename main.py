import torch
import clip
from PIL import Image
import os
import pickle
import numpy as np
import faiss

# Config
STORE_PATH = "image_store.pkl"
device = "cpu"  # change to "cuda" if GPU available

# Load CLIP model
model, preprocess = clip.load("ViT-B/16", device=device)

# In-memory store
paths = []
features_list = [] 
features_matrix = None
faiss_index = None

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
    added_any = False

    for path in input_paths:
        path = os.path.abspath(path)

        # If directory → iterate files
        if os.path.isdir(path):
            files = os.listdir(path)
            files = [os.path.join(path, f) for f in files]
        elif os.path.isfile(path):
            files = [path]
        else:
            print(f"Invalid path: {path}")
            continue

        for img_path in files:
            if not img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif')):
                continue

            img_path = os.path.abspath(img_path)

            if img_path in paths:
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
    global features_matrix, faiss_index

    if len(features_list) > 0:
        features_matrix = np.vstack(features_list).astype(np.float32)

        dim = features_matrix.shape[1]
        faiss_index = faiss.IndexFlatIP(dim)
        faiss_index.add(features_matrix)

    else:
        features_matrix = None
        faiss_index = None


# Cosine Similarity
def cosine_similarity(a, b):
    return np.dot(a, b)

# Search (Image → Image)
def image_search(query_image_path, top_k=3):
    if faiss_index is None:
        return []

    query = extract_image_features(query_image_path).reshape(1,-1)

    scores, indices  = faiss_index.search(query,top_k)

    return [(scores[0][i], paths[indices[0][i]]) for i in range(len(indices[0]))]

# Search (Text → Image)
def text_search(text, top_k=3):
   
    if faiss_index is None:
        return []

    query = extract_text_features(text).reshape(1,-1)
    scores, indices = faiss_index.search(query,top_k)

    return [(scores[0][i], paths[indices[0][i]]) for i in range(len(indices[0]))]


# Run
if __name__ == "__main__":
    load_store()
    add_images("images")

    print("Choose search type:")
    print("1. Image search")
    print("2. Text search")

    choice = input("Enter choice (1/2): ").strip()

    if choice == "1":
        print("\nImage search mode (type 'q' to exit)")
        while True:
            query_path = input("Enter image path: ").strip()

            if query_path.lower() == "q":
                print("Exiting...")
                break

            if not os.path.exists(query_path):
                print("Invalid image path")
                continue

            results = image_search(query_path, top_k=3)

            print("\nTop Matches:")
            for score, path in results:
                print(f"{path} -> {score:.4f}")

    elif choice == "2":
        print("\nText search mode (type 'q' to exit)")
        while True:
            query_text = input("Enter text query: ").strip()

            if query_text.lower() == "q":
                print("Exiting...")
                break

            results = text_search(query_text, top_k=3)

            print("\nTop Matches:")
            for score, path in results:
                print(f"{path} -> {score:.4f}")

    else:
        print("Invalid choice.")