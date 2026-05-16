import os
import store as store_module
from features import extract_image_features
from index import build_index
from search import image_search, text_search

def add_images(folder):
    if not os.path.exists(folder):
        print(f"Folder not found: {folder}")
        return

    existing_paths = {os.path.abspath(item["path"]) for item in store_module.store}
    added = 0

    for file in os.listdir(folder):
        path = os.path.abspath(os.path.join(folder, file))

        if not path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        if path in existing_paths:
            continue

        print(f"Adding image: {file}")

        try:
            features = extract_image_features(path)

            store_module.store.append({
                "path": path,
                "features": features
            })

            existing_paths.add(path)
            added += 1

        except Exception as e:
            print(f"Failed to process {file}: {e}")

    if added > 0:
        store_module.save_store()
        build_index()
        print(f"Added {added} new images")
    else:
        print("No new images added")


if __name__ == "__main__":
    store_module.load_store()

    images_folder = os.path.join(os.path.dirname(__file__), "images")
    add_images(images_folder)
    
    # Build index if needed
    if len(store_module.store) > 0:
        import index as index_module
        if index_module.faiss_index is None:
            build_index()

    print("1. Image search\n2. Text search")
    choice = input("Choice: ")

    if choice == "1":
        while True:
            try:
                query_path = input("Image path: ")
                results = image_search(query_path)
                
                if not results:
                    print("No results found")
                else:
                    for score, path in results:
                        print(f"{path} - {float(score):.4f}")
            except EOFError:
                break
            except Exception as e:
                print(f"Error: {e}")

    elif choice == "2":
        while True:
            try:
                query = input("Query: ")
                results = text_search(query)
                
                if not results:
                    print("No results found")
                else:
                    for score, path in results:
                        print(f"{path} - {float(score):.4f}")
            except EOFError:
                break
            except Exception as e:
                print(f"Error: {e}")