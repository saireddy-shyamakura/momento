import numpy as np
import faiss
import store as store_module

faiss_index = None
features_matrix = None


def build_index():
    global faiss_index, features_matrix

    if len(store_module.store) > 0:
        print(f"Building index for {len(store_module.store)} items")

        features_matrix = np.vstack(
            [item["features"] for item in store_module.store]
        ).astype(np.float32)

        dim = features_matrix.shape[1]

        # Inner Product (works as cosine similarity since normalized)
        faiss_index = faiss.IndexFlatIP(dim)
        faiss_index.add(features_matrix)

    else:
        faiss_index = None
        features_matrix = None