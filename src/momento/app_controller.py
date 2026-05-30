"""
app_controller.py — Main application orchestrator for Momento.

Coordinates the three-phase workflow:
1. Initialization (load or create index)
2. Auto-indexing (run all features on folder)
3. Interactive query interface

Handles graceful shutdown via SIGINT/SIGTERM signal handlers.
Supports checkpoint-based recovery for interrupted operations.
"""

import os
import sys
import json
from typing import Optional, Dict, List
from dataclasses import dataclass
from importlib.metadata import version as _pkg_version

from .logger import setup_logger, get_logger
from .index import Index
from .cache import clear_cache
from .config import BASE_DIR, CHROMA_DB_DIR, COMPOSITE_SEP, MAX_SEARCH_RESULTS, MomentoConfig, load_config
from .validation import validate_folder_path, validate_positive_int
from .indexer import Indexer, IndexingStats
from .query_manager import QueryManager
from .file_picker import FilePicker
from .lock import LockFile
from .search import image_search, text_search
from .shutdown import is_shutdown_requested, install_signal_handlers, reset_shutdown_flag
from .checkpoint import get_checkpoint_manager, FeatureStatus
from .storage_manager import get_storage_manager
from .features import clear_model_cache

logger = setup_logger(__name__)


@dataclass
class AppState:
    """Holds current application state."""
    current_folder: Optional[str] = None
    index: Optional[Index] = None
    last_indexing_stats: Optional[IndexingStats] = None
    is_first_run: bool = True
    config: Optional[MomentoConfig] = None


class AppController:
    """Main application controller orchestrating the full workflow."""
    
    def __init__(self, config: Optional[MomentoConfig] = None):
        """Initialize the app controller.
        
        Args:
            config: MomentoConfig instance. If None, loads from file/env/defaults.
        """
        self.state = AppState()
        self.state.config = config or load_config()
        lock_path = os.path.join(BASE_DIR, "momento.lock")
        self.lock_file = LockFile(lock_path)
        self.checkpoint_manager = get_checkpoint_manager()
        self.storage_manager = get_storage_manager()
        install_signal_handlers()
        
    def initialize_index(self) -> Index:
        """Load or create the index.
        
        Returns:
            Index instance ready for use
        """
        logger.info("Initializing index...")
        try:
            index = Index()
            logger.info(f"Index loaded: {index.get_vector_count()} vectors")
            self.state.index = index
            return index
        except Exception as e:
            logger.error(f"Failed to initialize index: {e}")
            raise
    
    def prompt_for_folder(self) -> str:
        """Prompt user for folder path.
        
        Returns:
            Valid folder path
        """
        picker = FilePicker()
        folder = picker.prompt_folder_path()
        
        # Validate folder
        is_valid, error_msg = validate_folder_path(folder)
        if not is_valid:
            logger.error(f"Invalid folder: {error_msg}")
            raise ValueError(error_msg)
        
        # Confirm before proceeding
        if not picker.confirm_folder(folder):
            print("Indexing cancelled.")
            sys.exit(0)
        
        self.state.current_folder = folder
        return folder
    
    def auto_index_folder(self, folder: str, reset_index: bool = True) -> IndexingStats:
        """Auto-index all features for a folder.
        
        Args:
            folder: Path to folder to index
            reset_index: If True, clear old index before starting
            
        Returns:
            IndexingStats with results
        """
        logger.info(f"Starting auto-index for: {folder}")
        
        # Reset index if specified
        if reset_index and self.state.index.get_vector_count() > 0:
            logger.info("Clearing previous index...")
            self.state.index.delete_all()
        
        # Run indexing
        indexer = Indexer(self.state.index)
        stats = indexer.index_all_features(folder)
        
        self.state.last_indexing_stats = stats
        logger.info(f"Indexing complete: {stats.total_vectors} vectors")
        
        return stats
    
    def run_query_interface(self, output_format: str = "text") -> None:
        """Run the interactive query interface.
        
        Args:
            output_format: 'text' or 'json' for result display
        """
        if not self.state.current_folder:
            logger.error("No folder selected for querying")
            return
        
        if not self.state.index or self.state.index.get_vector_count() == 0:
            logger.error("Index is empty. Please index a folder first.")
            return
        
        manager = QueryManager(self.state.index, self.state.current_folder)
        if output_format == "json":
            # In JSON mode, run a single search and output JSON results
            self._run_json_query(manager)
        else:
            manager.run_interactive_loop()
    
    def _run_json_query(self, manager: QueryManager) -> None:
        """Run a single query and output results as JSON.
        
        Args:
            manager: QueryManager instance
        """
        print("Momento JSON mode — enter a query path or text")
        print("Enter 'q' to quit")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() in ('q', 'quit', 'exit'):
                    break
                
                # Determine if it's a file path or text query
                if os.path.exists(query):
                    results = image_search(
                        query_image_path=query,
                        index=self.state.index,
                        top_k=MAX_SEARCH_RESULTS,
                        threshold=manager.threshold,
                        use_aggregation=manager.use_aggregation,
                    )
                else:
                    results = text_search(
                        text=query,
                        index=self.state.index,
                        top_k=MAX_SEARCH_RESULTS,
                        threshold=manager.threshold,
                        use_aggregation=manager.use_aggregation,
                    )
                
                output = {
                    "query": query,
                    "results": [
                        {"score": round(score, 4), "path": path}
                        for score, path in results
                    ],
                    "total": len(results),
                }
                print(json.dumps(output, indent=2))
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except EOFError:
                break
            except Exception as e:
                print(json.dumps({"error": str(e)}))
    
    def run_full_workflow(self) -> None:
        """Execute the complete three-phase workflow.
        
        Phase 1: Initialize index
        Phase 2: Prompt for folder and auto-index
        Phase 3: Run interactive query interface
        """
        reset_shutdown_flag()

        try:
            # Phase 1: Initialize
            print("\n" + "="*50)
            print("Momento - Semantic Search Engine")
            print("="*50)

            self.initialize_index()

            # Phase 2: Decide whether to index or use existing index
            print("\n--- Phase 2: Index or Search ---")
            index_count = self.state.index.get_vector_count() if self.state.index else 0

            if index_count > 0:
                print(f"Existing index detected: {index_count} vectors.")
                print("1. Search existing index")
                print("2. Index a new folder (replaces or appends)")
                print("q. Quit")
                choice = input("Choice (1-2, or 'q'): ").strip().lower()

                if choice == '1':
                    # Use existing index for searching
                    self.state.current_folder = os.getcwd()
                    print("\n--- Phase 3: Interactive Search ---")
                    self.run_query_interface()
                    return
                elif choice == '2':
                    # Prompt for folder and index as before
                    folder = self.prompt_for_folder()
                    stats = self.auto_index_folder(folder)
                    self._print_indexing_summary(stats)
                    print("\n--- Phase 3: Interactive Search ---")
                    self.run_query_interface()
                    return
                else:
                    print("Goodbye!")
                    return
            else:
                # No existing vectors — must index first
                print("No existing index found — please select a folder to index.")
                folder = self.prompt_for_folder()
                stats = self.auto_index_folder(folder)
                self._print_indexing_summary(stats)
                print("\n--- Phase 3: Interactive Search ---")
                self.run_query_interface()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            print(f"Error: {e}")
            sys.exit(1)
        finally:
            # Release CLIP model GPU/CPU memory before exiting
            clear_model_cache()
            # Ensure ChromaDB is cleanly closed on exit
            if self.state.index is not None:
                self.state.index.close()
    
    def _print_indexing_summary(self, stats: IndexingStats) -> None:
        """Pretty-print indexing statistics."""
        print("\n" + "="*50)
        print("Indexing Complete!")
        print("="*50)
        print(f"Images added:     {stats.images_added}")
        print(f"Videos added:     {stats.videos_added}")
        print(f"Objects added:    {stats.objects_added}")
        print(f"OCR entries:      {stats.ocr_added}")
        print(f"Total vectors:    {stats.total_vectors}")
        print(f"Time taken:       {stats.duration_secs:.2f}s")
        
        if stats.errors:
            print(f"\n⚠️  {len(stats.errors)} error(s) occurred:")
            for error in stats.errors[:5]:  # Show first 5 errors
                print(f"  - {error}")
            if len(stats.errors) > 5:
                print(f"  ... and {len(stats.errors) - 5} more")
        
        print("="*50)
    
    def handle_utility_flags(self, args) -> bool:
        """Handle utility CLI flags (--version, --reset, etc).
        
        Args:
            args: Parsed CLI arguments
            
        Returns:
            True if a utility flag was handled and app should exit
        """
        if hasattr(args, 'version') and args.version:
            try:
                print(f"momento {_pkg_version('momento')}")
            except Exception:
                print("momento 2.0.0")
            return True
        
        # Initialize index for utility operations
        index = self.initialize_index()
        
        if hasattr(args, 'reset') and args.reset:
            index.delete_all()
            print("Index reset: all entries deleted.")
            return True
        
        if hasattr(args, 'count') and args.count:
            print(index.get_vector_count())
            return True
        
        if hasattr(args, 'verify') and args.verify:
            self._verify_index(index)
            return True
        
        if hasattr(args, 'cache_clean') and args.cache_clean:
            clear_cache()
            return True
        
        return False
    
    def _verify_index(self, index: Index) -> int:
        """Remove stale entries whose file paths no longer exist.
        
        Args:
            index: Index to verify
            
        Returns:
            Count of removed stale entries
        """
        logger.info("Verifying index...")
        paths = index.get_all_paths()
        stale = [p for p in paths if not os.path.exists(p.split(COMPOSITE_SEP)[0])]
        
        if stale:
            index.delete_paths(stale)
            print(f"Removed {len(stale)} stale entries.")
        else:
            print("Index is clean — no stale entries found.")
        
        return len(stale)