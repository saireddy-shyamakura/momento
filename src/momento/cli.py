import os
import sys
import argparse
import subprocess
from typing import List, Tuple

from .logger import setup_logger
from .index import Index
from .search import image_search, text_search
from .validation import (
    validate_image_path, validate_text_query, validate_folder_path,
    validate_choice, validate_positive_int, ValidationError
)
from .add_images import add_images
from .config import SIMILARITY_THRESHOLD, BASE_DIR
from .output import render_result, open_file
from .lock import LockFile

logger = setup_logger(__name__)

try:
    from importlib.metadata import version as _pkg_version
    _VERSION = _pkg_version("momento")
except Exception:
    _VERSION = "1.0.0"


def verify_index(index: Index) -> int:
    """Remove stale entries whose file paths no longer exist on disk.

    Returns the count of removed (stale) entries.
    """
    paths = index.get_all_paths()
    stale = [p for p in paths if not os.path.exists(p)]
    if stale:
        index.delete_paths(stale)
        print(f"Removed {len(stale)} stale entries.")
    else:
        print("Index is clean — no stale entries found.")
    return len(stale)


def run_cli():
    """Main application entry point with input validation."""
    parser = argparse.ArgumentParser(description="Momento Image Search Engine")
    parser.add_argument("--dir", "-d", type=str, help="Directory containing images to index on startup")
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    parser.add_argument("--reset", action="store_true", help="Delete all index entries and exit")
    parser.add_argument("--count", action="store_true", help="Print number of indexed images and exit")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Similarity threshold for search results (0.0–1.0)")
    parser.add_argument("--verify", action="store_true", help="Remove stale index entries and exit")
    parser.add_argument("--open", action="store_true", help="Open the top search result in the system viewer")
    args = parser.parse_args()

    # --version: print version and exit 0 (no index needed)
    if args.version:
        print(f"momento {_VERSION}")
        sys.exit(0)

    # --threshold validation
    if args.threshold is not None:
        if not (0.0 <= args.threshold <= 1.0):
            print("Error: --threshold must be between 0.0 and 1.0", file=sys.stderr)
            sys.exit(1)
        threshold = args.threshold
    else:
        threshold = SIMILARITY_THRESHOLD

    try:
        # Initialize index
        index = Index()

        # Handle --reset flag
        if args.reset:
            index.delete_all()
            print("Index reset: all entries deleted.")
            sys.exit(0)

        # Handle --count flag
        if args.count:
            print(index.get_vector_count())
            sys.exit(0)

        # Handle --verify flag
        if args.verify:
            verify_index(index)
            sys.exit(0)

        # Add images from folder if provided
        if args.dir:
            images_folder = os.path.abspath(args.dir)
            logger.info(f"Indexing images from provided directory: {images_folder}")
            add_images(images_folder, index)
        else:
            # Default fallback for backwards compatibility
            images_folder = os.path.join(BASE_DIR, "..", "images")
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
                    image_search_mode(index, threshold, open_result=args.open)

                elif choice == "2":
                    text_search_mode(index, threshold, open_result=args.open)
                    
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
            print(render_result(j, score, path))
            
        if i + page_size < total:
            try:
                cont = input("\nPress Enter to see more results, or 'q' to quit: ").strip().lower()
                if cont == 'q':
                    break
            except KeyboardInterrupt:
                print() # Print newline to keep formatting clean
                break


def image_search_mode(index: Index, threshold: float = SIMILARITY_THRESHOLD, open_result: bool = False):
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
            results = image_search(query_path, index, top_k, threshold=threshold)
            
            print_paginated_results(results)
            if open_result and results:
                try:
                    open_file(results[0][1])
                except (OSError, subprocess.CalledProcessError) as e:
                    print(f"Error opening file: {e}")
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


def text_search_mode(index: Index, threshold: float = SIMILARITY_THRESHOLD, open_result: bool = False):
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
            results = text_search(query, index, top_k, threshold=threshold)
            
            print_paginated_results(results)
            if open_result and results:
                try:
                    open_file(results[0][1])
                except (OSError, subprocess.CalledProcessError) as e:
                    print(f"Error opening file: {e}")
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


def main():
    lock = LockFile(os.path.join(BASE_DIR, "momento.lock"))
    if not lock.acquire():
        print("Momento is already running in another process.", file=sys.stderr)
        sys.exit(1)
        
    try:
        run_cli()
    finally:
        lock.release()

if __name__ == "__main__":
    main()
