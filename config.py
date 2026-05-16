import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORE_PATH = os.path.join(BASE_DIR, "image_store.pkl")
DEVICE = "cpu"  # or "cuda"
MODEL_NAME = "ViT-B/16"