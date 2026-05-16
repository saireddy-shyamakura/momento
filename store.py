import os
import pickle
from config import STORE_PATH

store = []

def load_store():
    global store

    if os.path.exists(STORE_PATH):
        try:
            with open(STORE_PATH, "rb") as f:
                store = pickle.load(f)

            print(f"Loaded {len(store)} images")

        except Exception as e:
            print(f"Store corrupted, starting fresh: {e}")
            store = []
    else:
        store = []
        print("Starting fresh")


def save_store():
    temp_path = STORE_PATH + ".tmp"

    with open(temp_path, "wb") as f:
        pickle.dump(store, f)

    os.replace(temp_path, STORE_PATH)  # atomic write