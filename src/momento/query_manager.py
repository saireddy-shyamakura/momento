"""
query_manager.py — Interactive query interface for Momento.

Handles user interactions after indexing:
- Search menu (image search, text search)
- Result display and navigation
- File opening
- Re-indexing capability
"""

import os
import sys
from typing import List, Tuple, Optional

from .index import Index
from .search import image_search, text_search
from .validation import validate_image_path, validate_text_query, validate_choice
from .output import render_result, open_file
from .config import SIMILARITY_THRESHOLD, MAX_SEARCH_RESULTS
from .logger import get_logger

logger = get_logger(__name__)


class QueryManager:
    """Manages interactive post-indexing query interface."""
    
    def __init__(self, index: Index, current_folder: str, threshold: float = SIMILARITY_THRESHOLD):
        """Initialize query manager.
        
        Args:
            index: Index instance to search
            current_folder: Current working folder
            threshold: Similarity threshold for search results
        """
        self.index = index
        self.current_folder = current_folder
        self.threshold = threshold
        self.use_aggregation = True  # Use score aggregation for multi-vectors
    
    def show_menu(self) -> str:
        """Display search menu and get user choice.
        
        Returns:
            User's menu choice ('1', '2', 'q', etc.)
        """
        print("\n" + "="*50)
        print("Search Menu")
        print("="*50)
        print("1. 🖼️  Image search (find similar images)")
        print("2. 📝 Text search (search by description)")
        print("3. 🔧 Settings (change threshold)")
        print("4. 📂 Re-index different folder")
        print("5. ℹ️  Index info")
        print("q. Quit")
        print("="*50)
        
        while True:
            choice = input("Choose (1-5, or 'q'): ").strip().lower()
            
            is_valid, error_msg = validate_choice(choice, ["1", "2", "3", "4", "5", "q"])
            if not is_valid:
                print(f"Invalid input: {error_msg}. Try again.")
                continue
            
            return choice
    
    def image_search(self) -> None:
        """Interactive image search mode."""
        print("\n📷 Image Search")
        print("-" * 40)
        image_path = input("Enter image path to search: ").strip()
        
        # Validate path
        is_valid, error_msg = validate_image_path(image_path)
        if not is_valid:
            print(f"Error: {error_msg}")
            return
        
        print("Searching...", end=" ", flush=True)
        try:
            results = image_search(
                image_path,
                self.index,
                top_k=MAX_SEARCH_RESULTS,
                threshold=self.threshold,
                use_aggregation=self.use_aggregation
            )
            print("✓\n")
            self._display_results(results)
        except Exception as e:
            logger.error(f"Image search failed: {e}")
            print(f"✗\nError: {e}")
    
    def text_search(self) -> None:
        """Interactive text search mode."""
        print("\n📝 Text Search")
        print("-" * 40)
        query_text = input("Enter search description: ").strip()
        
        # Validate query
        is_valid, error_msg = validate_text_query(query_text)
        if not is_valid:
            print(f"Error: {error_msg}")
            return
        
        print("Searching...", end=" ", flush=True)
        try:
            results = text_search(
                query_text,
                self.index,
                top_k=MAX_SEARCH_RESULTS,
                threshold=self.threshold,
                use_aggregation=self.use_aggregation
            )
            print("✓\n")
            self._display_results(results)
        except Exception as e:
            logger.error(f"Text search failed: {e}")
            print(f"✗\nError: {e}")
    
    def change_threshold(self) -> None:
        """Interactively change similarity threshold."""
        print("\n🔧 Similarity Threshold")
        print("-" * 40)
        print(f"Current threshold: {self.threshold:.2f}")
        print("(Valid range: 0.0 - 1.0, default: 0.20)")
        
        while True:
            try:
                value = input("Enter new threshold: ").strip()
                if value.lower() == 'q':
                    return
                
                threshold = float(value)
                if not (0.0 <= threshold <= 1.0):
                    print("Please enter a value between 0.0 and 1.0")
                    continue
                
                self.threshold = threshold
                print(f"✓ Threshold updated to {threshold:.2f}")
                break
            except ValueError:
                print("Invalid number. Try again or enter 'q' to cancel.")
    
    def show_index_info(self) -> None:
        """Display index statistics."""
        print("\n" + "="*50)
        print("Index Information")
        print("="*50)
        
        vector_count = self.index.get_vector_count()
        print(f"Total vectors indexed:  {vector_count}")
        print(f"Current folder:         {self.current_folder}")
        print(f"Similarity threshold:   {self.threshold:.2f}")
        print(f"Aggregation mode:       {'Enabled' if self.use_aggregation else 'Disabled'}")
        
        print("="*50)
    
    def _display_results(self, results: List[Tuple[float, str]]) -> None:
        """Display search results with pagination.
        
        Args:
            results: List of (score, path) tuples
        """
        if not results:
            print("❌ No matches found above the threshold.")
            return
        
        print(f"✓ Found {len(results)} matches:\n")
        
        page_size = 5
        total = len(results)
        current_page = 0
        
        while True:
            start_idx = current_page * page_size
            end_idx = min(start_idx + page_size, total)
            
            # Display current page
            for i in range(start_idx, end_idx):
                score, path = results[i]
                result_num = i + 1
                print(render_result(result_num, score, path))
            
            # Navigation
            has_prev = current_page > 0
            has_next = end_idx < total
            
            nav_opts = []
            if has_prev:
                nav_opts.append("'p' for previous")
            if has_next:
                nav_opts.append("'n' for next")
            nav_opts.extend(["'o' to open result", "'q' to quit"])
            
            print(f"\n--- Page {current_page + 1}/{(total + page_size - 1) // page_size} ---")
            print(f"Options: {', '.join(nav_opts)}")
            
            choice = input("Choice: ").strip().lower()
            
            if choice == 'q':
                break
            elif choice == 'n' and has_next:
                current_page += 1
            elif choice == 'p' and has_prev:
                current_page -= 1
            elif choice == 'o':
                self._handle_open_result(results, start_idx, end_idx)
            else:
                print("Invalid choice.")
                continue
            
            print()
    
    def _handle_open_result(self, results: List[Tuple[float, str]], page_start: int, page_end: int) -> None:
        """Handle opening a result file.
        
        Args:
            results: All search results
            page_start: Start index of current page
            page_end: End index of current page
        """
        try:
            choice = int(input("  Enter result number to open: ")) - 1
            
            if not (page_start <= choice < page_end):
                print("  Invalid result number.")
                return
            
            score, path = results[choice]
            from .config import COMPOSITE_SEP
            # Handle composite IDs (path|||suffix → path)
            if COMPOSITE_SEP in path:
                base_path = path.split(COMPOSITE_SEP)[0]
            else:
                base_path = path
            
            if os.path.exists(base_path):
                open_file(base_path)
            else:
                print(f"  File not found: {base_path}")
        except (ValueError, IndexError):
            print("  Invalid input.")
    
    def run_interactive_loop(self) -> None:
        """Run the main interactive search loop."""
        print(f"\n✓ Ready to search! (Indexed: {self.index.get_vector_count()} vectors)")
        
        while True:
            try:
                choice = self.show_menu()
                
                if choice == "q":
                    print("\nGoodbye!")
                    break
                elif choice == "1":
                    self.image_search()
                elif choice == "2":
                    self.text_search()
                elif choice == "3":
                    self.change_threshold()
                elif choice == "4":
                    print("\nTo re-index a different folder, please restart the application.")
                    break
                elif choice == "5":
                    self.show_index_info()
            
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except EOFError:
                print("\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                print(f"Error: {e}")
