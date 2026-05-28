"""
cli.py — Command-line interface for Momento.

Entry point for the semantic search engine. Handles CLI arguments and
orchestrates the application workflow through AppController.

Supports Phase 2 commands:
- --log-format json|text
- config show, config set
- doctor, stats, benchmark

Supports Phase 5 commands:
- --quiet / --verbose / --debug for log level control
- --output json for machine-readable search results
- --dry-run for scanning without indexing
- --exclude for glob pattern exclusions
- export / import for index data portability
"""

import sys
import os
import json
import argparse

from .logger import setup_logger, set_log_format
from .app_controller import AppController
from .config import BASE_DIR, load_config, save_config, MomentoConfig
from .lock import LockFile

logger = setup_logger(__name__)

try:
    from importlib.metadata import version as _pkg_version
    _VERSION = _pkg_version("momento")
except Exception:
    _VERSION = "2.0.0"


def parse_arguments():
    """Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Momento — Multi-Modal Semantic Search Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start interactive search (prompted for folder)
  momento

  # Index a specific folder and start search
  momento --dir ~/Pictures

  # Show version
  momento --version

  # Reset all indexed data
  momento --reset

  # Count indexed vectors
  momento --count

  # Verify index and remove stale entries
  momento --verify

  # Show system health
  momento doctor

  # Show index statistics
  momento stats

  # Run performance benchmark
  momento benchmark

  # View/edit configuration
  momento config show
  momento config set threshold 0.30

  # Structured JSON logging
  momento --log-format json

  # Dry-run: scan folder without indexing
  momento --dir ~/Pictures --dry-run

  # Exclude certain file patterns
  momento --dir ~/Pictures --exclude "*.txt,private/"

  # Export all index data
  momento export --format npz -o export.npz

  # Import index data
  momento import --from export.npz

Note: All features (multi-embedding, video, YOLO, OCR) are enabled by default.
        """
    )

    # ── Utility flags ────────────────────────────────────────────────
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print version and exit"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete all indexed entries and exit"
    )
    parser.add_argument(
        "--count",
        action="store_true",
        help="Print number of indexed vectors and exit"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Remove stale index entries and exit"
    )
    parser.add_argument(
        "--cache-clean",
        action="store_true",
        help="Delete all cached embeddings to free disk space"
    )

    # ── Logging format and level ─────────────────────────────────────
    parser.add_argument(
        "--log-format",
        type=str,
        default=None,
        choices=["text", "json"],
        help="Log output format (text or json)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG level)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-essential output (WARNING level)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging (DEBUG level, most verbose)"
    )

    # ── Workflow flags ───────────────────────────────────────────────
    parser.add_argument(
        "--dir", "-d",
        type=str,
        help="Directory to index on startup (optional)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Similarity threshold for search results (0.0-1.0, default: 0.20)"
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open top search result in system viewer"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        choices=["text", "json"],
        help="Output format for search results (text or json)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan folder and show what would be indexed, then exit"
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default=None,
        help="Comma-separated glob patterns to exclude (e.g., '*.txt,private/')"
    )

    # ── Subcommands ──────────────────────────────────────────────────
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # config subcommand
    config_parser = subparsers.add_parser("config", help="View or modify configuration")
    config_subparsers = config_parser.add_subparsers(dest="config_action", help="Config actions")
    config_subparsers.add_parser("show", help="Show current configuration")
    config_set = config_subparsers.add_parser("set", help="Set a configuration value")
    config_set.add_argument("key", type=str, help="Configuration key (e.g., threshold)")
    config_set.add_argument("value", type=str, help="Configuration value")

    # doctor subcommand
    subparsers.add_parser("doctor", help="Run system health check")

    # stats subcommand
    subparsers.add_parser("stats", help="Show index statistics")

    # benchmark subcommand
    subparsers.add_parser("benchmark", help="Run performance benchmarks")

    # export subcommand
    export_parser = subparsers.add_parser("export", help="Export index data")
    export_parser.add_argument("--format", type=str, default="npz", choices=["npz", "json"],
                               help="Export format (default: npz)")
    export_parser.add_argument("-o", "--output", type=str, default=None,
                               help="Output file path (default: momento_export.npz)")

    # import subcommand
    import_parser = subparsers.add_parser("import", help="Import index data")
    import_parser.add_argument("--from", dest="from_file", type=str, required=True,
                               help="File to import from")

    # Deprecated flags (for backward compatibility)
    parser.add_argument(
        "--multi-embed",
        action="store_true",
        help="[DEPRECATED] Features are now always enabled"
    )
    parser.add_argument(
        "--include-video",
        action="store_true",
        help="[DEPRECATED] Features are now always enabled"
    )
    parser.add_argument(
        "--yolo",
        action="store_true",
        help="[DEPRECATED] Features are now always enabled"
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="[DEPRECATED] Features are now always enabled"
    )
    parser.add_argument(
        "--all-features",
        action="store_true",
        help="[DEPRECATED] Features are now always enabled"
    )

    return parser.parse_args()


def validate_threshold(threshold: float) -> bool:
    """Validate similarity threshold.

    Args:
        threshold: Threshold value to validate

    Returns:
        True if valid
    """
    if not (0.0 <= threshold <= 1.0):
        print("Error: --threshold must be between 0.0 and 1.0", file=sys.stderr)
        return False
    return True


def show_deprecation_warnings(args) -> None:
    """Show deprecation warnings for old flags.

    Args:
        args: Parsed arguments
    """
    deprecated_flags = [
        ('multi_embed', '--multi-embed'),
        ('include_video', '--include-video'),
        ('yolo', '--yolo'),
        ('ocr', '--ocr'),
        ('all_features', '--all-features'),
    ]

    for attr, flag_name in deprecated_flags:
        if hasattr(args, attr) and getattr(args, attr):
            print(f"ℹ️  Note: {flag_name} is deprecated (features are auto-enabled)")


def _handle_config_command(args) -> None:
    """Handle the 'config' subcommand.

    Args:
        args: Parsed arguments with config_action, key, value.
    """
    config = load_config()

    if args.config_action == "show":
        from .config import CONFIG_FILE
        print(f"\nConfiguration file: {CONFIG_FILE}")
        print("=" * 50)
        print(f"enable_multi_embed       = {config.enable_multi_embed}")
        print(f"enable_video_indexing    = {config.enable_video_indexing}")
        print(f"enable_yolo              = {config.enable_yolo}")
        print(f"enable_ocr               = {config.enable_ocr}")
        print(f"similarity_threshold     = {config.similarity_threshold}")
        print(f"max_search_results       = {config.max_search_results}")
        print(f"indexing_batch_size      = {config.indexing_batch_size}")
        print(f"progress_bar_enabled     = {config.progress_bar_enabled}")
        print(f"video_frame_interval     = {config.video_frame_interval}")
        print(f"max_frames_per_video     = {config.max_frames_per_video}")
        print(f"yolo_model               = {config.yolo_model}")
        print(f"yolo_confidence_threshold = {config.yolo_confidence_threshold}")
        print(f"ocr_languages            = {config.ocr_languages}")
        print(f"ocr_min_text_length      = {config.ocr_min_text_length}")
        print(f"augmentation_count       = {config.augmentation_count}")
        print(f"cache_max_size_gb        = {config.cache_max_size_gb}")
        print(f"log_format               = {config.log_format}")
        print(f"log_level                = {config.log_level}")
        print("=" * 50)

    elif args.config_action == "set":
        key = args.key
        value = args.value

        # Map config key to dataclass attribute and type
        _TYPE_MAP: dict = {
            "enable_multi_embed": bool,
            "enable_video_indexing": bool,
            "enable_yolo": bool,
            "enable_ocr": bool,
            "similarity_threshold": float,
            "max_search_results": int,
            "indexing_batch_size": int,
            "progress_bar_enabled": bool,
            "video_frame_interval": float,
            "max_frames_per_video": int,
            "yolo_model": str,
            "yolo_confidence_threshold": float,
            "ocr_languages": list,
            "ocr_min_text_length": int,
            "augmentation_count": int,
            "cache_max_size_gb": int,
            "log_format": str,
            "log_level": str,
        }

        if key not in _TYPE_MAP:
            print(f"Error: Unknown config key '{key}'")
            print(f"Valid keys: {', '.join(sorted(_TYPE_MAP.keys()))}")
            sys.exit(1)

        if not hasattr(config, key):
            print(f"Error: Unknown config key '{key}'")
            sys.exit(1)

        # Parse value
        try:
            target_type = _TYPE_MAP[key]
            if target_type == bool:
                parsed = value.lower() in ("true", "1", "yes")
            elif target_type == list:
                parsed = [v.strip() for v in value.split(",")]
            else:
                parsed = target_type(value)
        except (ValueError, TypeError) as e:
            print(f"Error: Cannot parse '{value}' as {target_type.__name__}: {e}")
            sys.exit(1)

        setattr(config, key, parsed)
        save_config(config)
        print(f"✓ Config updated: {key} = {parsed}")


def _handle_doctor_command() -> None:
    """Handle the 'doctor' subcommand."""
    from .diagnostics import run_doctor, print_doctor_report
    result = run_doctor()
    print_doctor_report(result)


def _handle_stats_command() -> None:
    """Handle the 'stats' subcommand."""
    from .diagnostics import get_index_stats, print_index_stats
    from .config import CHROMA_DB_DIR
    stats = get_index_stats(CHROMA_DB_DIR)
    print_index_stats(stats)


def _handle_benchmark_command() -> None:
    """Handle the 'benchmark' subcommand."""
    from .diagnostics import run_benchmark, print_benchmark_report
    from .config import CHROMA_DB_DIR
    result = run_benchmark(CHROMA_DB_DIR)
    print_benchmark_report(result)


def _handle_export_command(args) -> None:
    """Handle the 'export' subcommand.

    Exports all index data (ids, embeddings, metadatas) to a file.

    Args:
        args: Parsed arguments with format and output.
    """
    from .index import Index
    from .config import CHROMA_DB_DIR

    output_path = args.output or "momento_export.npz"
    export_format = args.format

    print(f"📦 Exporting index data to {output_path}...")
    try:
        index = Index(db_path=CHROMA_DB_DIR)
        ids, embeddings, metadatas = index.export_all_data()

        if not ids:
            print("❌ Index is empty — nothing to export.")
            return

        if export_format == "npz":
            import numpy as np
            np.savez_compressed(
                output_path,
                ids=ids,
                embeddings=np.array(embeddings, dtype=np.float32),
                metadatas=metadatas,
            )
        elif export_format == "json":
            export_data = {
                "ids": ids,
                "embeddings": embeddings,
                "metadatas": metadatas,
                "version": "2.0.0",
            }
            with open(output_path, "w") as f:
                json.dump(export_data, f, indent=2, default=str)

        print(f"✓ Exported {len(ids)} vectors to {output_path}")
    except Exception as e:
        print(f"❌ Export failed: {e}")
        sys.exit(1)


def _handle_import_command(args) -> None:
    """Handle the 'import' subcommand.

    Imports index data from a file.

    Args:
        args: Parsed arguments with from_file.
    """
    from .index import Index
    from .config import CHROMA_DB_DIR

    input_path = args.from_file
    if not os.path.exists(input_path):
        print(f"❌ Import file not found: {input_path}")
        sys.exit(1)

    print(f"📦 Importing index data from {input_path}...")
    try:
        index = Index(db_path=CHROMA_DB_DIR)

        if input_path.endswith(".npz"):
            import numpy as np
            data = np.load(input_path, allow_pickle=True)
            ids = data.get("ids")
            embeddings_data = data.get("embeddings")
            metadatas_data = data.get("metadatas")

            if ids is None or embeddings_data is None:
                print("❌ Invalid export file: missing ids or embeddings.")
                sys.exit(1)

            ids_list = ids.tolist() if hasattr(ids, 'tolist') else list(ids)
            embeddings_list = embeddings_data.tolist() if hasattr(embeddings_data, 'tolist') else list(embeddings_data)
            metadatas_list = metadatas_data.tolist() if metadatas_data is not None else [{}] * len(ids_list)

        elif input_path.endswith(".json"):
            with open(input_path, "r") as f:
                data = json.load(f)
            ids_list = data.get("ids", [])
            embeddings_list = data.get("embeddings", [])
            metadatas_list = data.get("metadatas", [{}] * len(ids_list))
        else:
            print(f"❌ Unsupported format: {input_path}. Use .npz or .json files.")
            sys.exit(1)

        if not ids_list:
            print("❌ No data to import.")
            return

        count = index.import_all_data(ids_list, embeddings_list, metadatas_list)
        print(f"✓ Imported {count} vectors into index")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        sys.exit(1)


def _handle_dry_run(args) -> None:
    """Handle --dry-run: scan folder and show what would be indexed.

    Args:
        args: Parsed arguments with dir, exclude.
    """
    from .validation import validate_folder_path
    from .file_picker import FilePicker

    folder = args.dir
    if not folder:
        print("❌ --dry-run requires --dir to specify a folder.")
        sys.exit(1)

    is_valid, error_msg = validate_folder_path(folder)
    if not is_valid:
        print(f"❌ Invalid folder: {error_msg}")
        sys.exit(1)

    picker = FilePicker()
    preview = picker.preview_indexable_files(folder)

    print(f"\n📋 Dry-Run Preview for: {folder}")
    print("=" * 50)
    print(f"Images:      {preview['image_count']}")
    print(f"Videos:      {preview['video_count']}")
    print(f"Total items: {preview['total_count']}")
    print("=" * 50)

    # Check exclude patterns
    if args.exclude:
        patterns = [p.strip() for p in args.exclude.split(",")]
        print(f"\n🔍 Exclude patterns: {', '.join(patterns)}")

    print("\n✅ Dry-run complete — no changes made.\n")


def run_cli():
    """Main CLI entry point."""
    import logging
    args = parse_arguments()

    # Apply log level flags
    if args.debug or args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        print("🔍 Debug mode enabled")
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    # Show deprecation warnings
    show_deprecation_warnings(args)

    # ── Handle subcommands that don't need lock or controller ────────
    if args.command == "doctor":
        _handle_doctor_command()
        return
    if args.command == "stats":
        _handle_stats_command()
        return
    if args.command == "benchmark":
        _handle_benchmark_command()
        return
    if args.command == "config":
        _handle_config_command(args)
        return
    if args.command == "export":
        _handle_export_command(args)
        return
    if args.command == "import":
        _handle_import_command(args)
        return

    # Handle --dry-run (doesn't need controller)
    if args.dry_run:
        _handle_dry_run(args)
        return

    # Initialize controller for main workflow
    controller = AppController()

    try:
        # Handle utility flags first (--version, --reset, etc.)
        if controller.handle_utility_flags(args):
            return

        # Validate threshold if provided
        if args.threshold is not None:
            if not validate_threshold(args.threshold):
                sys.exit(1)

        # Run main workflow
        if args.dir:
            # Legacy mode: --dir specified
            print(f"\n📂 Using folder: {args.dir}")
            controller.initialize_index()

            # Validate and confirm folder
            from .validation import validate_folder_path
            from .file_picker import FilePicker

            is_valid, error_msg = validate_folder_path(args.dir)
            if not is_valid:
                print(f"Error: {error_msg}")
                sys.exit(1)

            picker = FilePicker()
            if not picker.confirm_folder(args.dir):
                sys.exit(0)

            stats = controller.auto_index_folder(args.dir)
            controller._print_indexing_summary(stats)

            # Query interface
            controller.state.current_folder = args.dir
            controller.run_query_interface(output_format=args.output or "text")
        else:
            # New interactive mode
            controller.run_full_workflow()

    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"Error: {e}")
        sys.exit(1)


def main():
    """Main entry point with lock file management."""
    # Apply log format before anything else
    args, _ = argparse.ArgumentParser(add_help=False).parse_known_args()
    if hasattr(args, 'log_format') and args.log_format:
        try:
            set_log_format(args.log_format)
        except ValueError:
            pass

    # Subcommands that don't need a lock
    if hasattr(args, 'command') and args.command in (
        "doctor", "stats", "benchmark", "config", "export", "import"
    ):
        try:
            run_cli()
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
        return

    # --dry-run doesn't need a lock either
    if hasattr(args, 'dry_run') and args.dry_run:
        try:
            run_cli()
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
        return

    lock_path = os.path.join(BASE_DIR, "momento.lock")
    lock = LockFile(lock_path)

    if not lock.acquire():
        print("Momento is already running in another process.", file=sys.stderr)
        sys.exit(1)

    try:
        run_cli()
    finally:
        lock.release()


if __name__ == "__main__":
    main()