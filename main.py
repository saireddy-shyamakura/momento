import os
from logger import setup_logger
import argparse
from typing import List, Tuple
from features import extract_image_features
from index import Index
from search import image_search, text_search
from validation import (
    validate_image_path, validate_text_query, validate_folder_path,
    validate_choice, validate_positive_int, ValidationError
)
from add_images import add_images

logger = setup_logger(__name__)


def main():
    """Main application entry point with input validation."""
    parser = argparse.ArgumentParser(description="Momento Image Search Engine")
    parser.add_argument("--dir", "-d", type=str, help="Directory containing images to index on startup")
    args = parser.parse_args()

    try:
        # Initialize index
        index = Index()
        
        # Add images from folder if provided
        if args.dir:
            images_folder = os.path.abspath(args.dir)
            logger.info(f"Indexing images from provided directory: {images_folder}")
            add_images(images_folder, index)
        else:
            # Default fallback for backwards compatibility
            images_folder = os.path.join(os.path.dirname(__file__), "images")
            if os.path.exists(images_folder):
                add_images(images_folder, index)
        
        logger.info(f"Database ready: {index.get_vector_count()} images indexed")

        print("\n=== Image Search Engine ===")
        print("1. Image search (find similar images)")
        print("2. Text search (search by description)")
        print("3. Index new images from directory")
        
        while True:
            try:
                choice = input("\nChoice (1, 2, 3, or 'q' to quit): ").strip()
                
                # Validate choice
                is_valid, error_msg = validate_choice(choice, ["1", "2", "3", "q"])
                if not is_valid:
                    print(f"Invalid input: {error_msg}")
                    continue
                
                if choice == "q":
                    print("Goodbye!")
                    break

                if choice == "1":
                    image_search_mode(index)

                elif choice == "2":
                    text_search_mode(index)
                    
                elif choice == "3":
                    add_images_mode(index)

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except EOFError:
                print("\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                print(f"Error: {e}")

    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        raise


def print_paginated_results(results: List[Tuple[float, str]], page_size: int = 10):
    """Print search results with simple pagination."""
    if not results:
        print("\nNo relevant results found (all matches were below the similarity threshold).")
        return
        
    total = len(results)
    print(f"\nFound {total} matches:")
    
    for i in range(0, total, page_size):
        chunk = results[i:i+page_size]
        for j, (score, path) in enumerate(chunk, i + 1):
            print(f"{j}. {path} -> {float(score):.4f}")
            
        if i + page_size < total:
            try:
                cont = input("\nPress Enter to see more results, or 'q' to quit: ").strip().lower()
                if cont == 'q':
                    break
            except KeyboardInterrupt:
                print() # Print newline to keep formatting clean
                break


def image_search_mode(index: Index):
    """Interactive image search mode."""
    if index.get_vector_count() == 0:
        print("No images in database. Add images first.")
        return
    
    print("\n--- Image Search Mode ---")
    while True:
        try:
            query_path = input("Image path (or 'back' to return): ").strip()
            
            if query_path.lower() == "back":
                break
            
            # Validate image path
            is_valid, error_msg = validate_image_path(query_path)
            if not is_valid:
                print(f"Invalid image: {error_msg}")
                continue
            
            # Validate top_k
            top_k_str = input("Number of results (default 3): ").strip()
            top_k = 3
            if top_k_str:
                try:
                    top_k = int(top_k_str)
                    is_valid, error_msg = validate_positive_int(top_k, "top_k")
                    if not is_valid:
                        print(f"Invalid input: {error_msg}")
                        continue
                except ValueError:
                    print("Please enter a valid number")
                    continue
            
            # Perform search
            results = image_search(query_path, index, top_k)
            
            print_paginated_results(results)
        except ValidationError as e:
            print(f"Validation error: {e}")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            break
        except Exception as e:
            logger.error(f"Search error: {e}")
            print(f"Error: {e}")


def text_search_mode(index: Index):
    """Interactive text search mode."""
    if index.get_vector_count() == 0:
        print("No images in database. Add images first.")
        return
    
    print("\n--- Text Search Mode ---")
    while True:
        try:
            query = input("Search query (or 'back' to return): ").strip()
            
            if query.lower() == "back":
                break
            
            # Validate text query
            is_valid, error_msg = validate_text_query(query)
            if not is_valid:
                print(f"Invalid query: {error_msg}")
                continue
            
            # Validate top_k
            top_k_str = input("Number of results (default 3): ").strip()
            top_k = 3
            if top_k_str:
                try:
                    top_k = int(top_k_str)
                    is_valid, error_msg = validate_positive_int(top_k, "top_k")
                    if not is_valid:
                        print(f"Invalid input: {error_msg}")
                        continue
                except ValueError:
                    print("Please enter a valid number")
                    continue
            
            # Perform search
            results = text_search(query, index, top_k)
            
            print_paginated_results(results)
        except ValidationError as e:
            print(f"Validation error: {e}")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            break
        except Exception as e:
            logger.error(f"Search error: {e}")
            print(f"Error: {e}")


def add_images_mode(index: Index):
    """Interactive mode to add images from a directory."""
    print("\n--- Index New Images ---")
    while True:
        try:
            folder_path = input("Directory path (or 'back' to return): ").strip()
            
            if folder_path.lower() == "back":
                break
                
            is_valid, error_msg = validate_folder_path(folder_path)
            if not is_valid:
                print(f"Invalid directory: {error_msg}")
                continue
                
            add_images(folder_path, index)
            print(f"Current database size: {index.get_vector_count()} images.")
            break
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            break
        except Exception as e:
            logger.error(f"Error adding images: {e}")
            print(f"Error: {e}")


if __name__ == "__main__":
    main()