from features import extract_image_features, extract_text_features
import index
import store as store_module

def _search(query_vector, top_k):
    if index.faiss_index is None:
        return []

    scores, indices = index.faiss_index.search(query_vector, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1 or idx >= len(store_module.store):
            continue

        results.append((scores[0][i], store_module.store[idx]["path"]))

    return results

def image_search(query_image_path, top_k=3):
    query = extract_image_features(query_image_path).reshape(1, -1)
    return _search(query, top_k)


def text_search(text, top_k=3):
    text = text.strip()
    text = f"a photo of {text}"

    query = extract_text_features(text).reshape(1, -1)

    return _search(query, top_k)