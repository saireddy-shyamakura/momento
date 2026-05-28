# Momento

A local, multi-modal semantic search engine built on **OpenAI CLIP** and **ChromaDB**. Momento enables powerful image and video search entirely on your machine, with no cloud dependencies.

## Features

- **Image-to-Image Search** — find visually similar images from a query image
- **Text-to-Image Search** — describe a scene and find matching media
- **Multi-Embedding Augmentation** — index 5 augmented views per image (flip, crop, brightness, contrast, rotate) for better recall
- **Video Keyframe Indexing** — extract and index frames from video files
- **YOLO Object Detection** — detect and embed individual object crops for fine-grained search
- **OCR Text Extraction** — extract on-image text and index it for text-based search
- **Parallel Execution** — Videos, Objects, and OCR features run concurrently for 2-4x faster indexing
- **Automatic Device Selection** — CUDA (NVIDIA), MPS (Apple Silicon), or CPU with OOM fallback
- **Crash Recovery** — checkpoint/resume so interrupted indexing picks up where it left off
- **Graceful Shutdown** — SIGINT/SIGTERM handlers complete the current batch before exiting
- **Structured Logging** — JSON log format with timing and event tracking
- **Embedding Cache** — LRU-evicted cache to speed up re-indexing

## Requirements

- Python 3.12+
- `torch` + `torchvision`
- `clip` (OpenAI CLIP)
- `chromadb` (vector store)
- `Pillow`
- `platformdirs`
- `tenacity` (retry logic)
- `psutil` (memory monitoring)

Optional dependencies (enable extra features):

| Dependency | Feature | Install |
|-----------|---------|---------|
| `opencv-python-headless` | Video keyframe extraction | `pip install opencv-python-headless` |
| `ultralytics` | YOLO object detection | `pip install ultralytics` |
| `easyocr` | OCR text extraction | `pip install easyocr` |
| `tqdm` | Progress bars during indexing | `pip install tqdm` |

## Installation

### Using uv (recommended)

```bash
pip install uv
git clone https://github.com/your-username/momento.git
cd momento
uv sync
```

### Using pip

```bash
pip install .
```

## Quick Start

Index your photo library and start searching:

```bash
momento --dir ~/Pictures
```

Start the interactive menu without auto-indexing:

```bash
momento
```

## CLI Reference

### Main Commands

| Command | Description |
|---------|-------------|
| `momento` | Interactive menu — index or search |
| `momento --dir PATH` | Index a folder and start interactive search |
| `momento doctor` | Run system health check |
| `momento stats` | Show detailed index statistics |
| `momento benchmark` | Run performance benchmarks |
| `momento export --format npz -o export.npz` | Export all index data |
| `momento import --from export.npz` | Import index data from file |

### Utility Flags

| Flag | Description |
|------|-------------|
| `--version` | Print version and exit |
| `--reset` | Delete all indexed entries and exit |
| `--count` | Print number of indexed vectors and exit |
| `--verify` | Remove stale entries whose source files no longer exist |
| `--cache-clean` | Delete all cached embeddings to free disk space |

### Workflow Flags

| Flag | Description |
|------|-------------|
| `--dir`, `-d PATH` | Directory containing media to index |
| `--threshold FLOAT` | Similarity threshold (0.0–1.0, default 0.20) |
| `--open` | Open top search result in system viewer |
| `--output json` | Output results as JSON (machine-readable) |
| `--dry-run` | Scan folder and show what would be indexed, then exit |
| `--exclude "*.txt,private/"` | Comma-separated glob patterns to exclude |

### Logging Flags

| Flag | Description |
|------|-------------|
| `--log-format text\|json` | Log output format (default: text) |
| `--quiet` | Suppress non-essential output (WARNING level) |
| `--verbose` | Enable verbose logging (DEBUG level) |
| `--debug` | Enable debug logging (most verbose) |

### Deprecated Flags

The following flags from v1 are deprecated. All features are now enabled by default:
`--multi-embed`, `--include-video`, `--yolo`, `--ocr`, `--all-features`

### Configuration

Momento automatically selects the best device:

| Hardware | Device |
|----------|--------|
| NVIDIA GPU | `cuda` |
| Apple Silicon | `mps` |
| Otherwise | `cpu` |

Override with `MOMENTO_DEVICE`:

```bash
MOMENTO_DEVICE=cpu momento --dir ~/Pictures
```

### Subcommands

#### `momento config show`

Display the current configuration.

#### `momento config set <key> <value>`

Update a configuration value and save it to `~/.config/momento/config.toml`.

Example:
```bash
momento config set threshold 0.30
momento config set yolo_model yolov8m.pt
```

#### `momento doctor`

Run a comprehensive system health check covering:
- Python and Momento versions
- Device (CUDA/MPS/CPU)
- CLIP model availability
- ChromaDB accessibility
- Disk space
- GPU availability
- Required and optional dependency status

#### `momento stats`

Display detailed index statistics:
- Total vectors and entries
- Per-type breakdown (images, videos, objects, OCR)
- Database size on disk
- Database path

#### `momento benchmark`

Run performance benchmarks:
- Embedding extraction latency
- Search latency (requires non-empty index)
- Batch throughput estimate

#### `momento export`

Export all index data to a file for backup or migration:

```bash
momento export --format npz -o my_index_backup.npz
momento export --format json -o my_index_backup.json
```

#### `momento import`

Import index data from a previously exported file:

```bash
momento import --from my_index_backup.npz
```

## Supported File Formats

- **Images**: `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`
- **Videos**: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.flv`

## Architecture

```
src/momento/
├── __init__.py         # Package metadata and version
├── app_controller.py   # Main application orchestrator
├── cli.py              # Command-line interface
├── config.py           # Central configuration with TOML support
├── index.py            # ChromaDB vector store wrapper
├── indexer.py          # Indexing orchestrator (parallel execution)
├── features.py         # CLIP model and feature extraction
├── add_images.py       # Image ingestion pipeline with cache
├── augment.py          # Image augmentation transforms
├── video.py            # Video keyframe extraction
├── yolo.py             # YOLO object detection
├── ocr.py              # OCR text extraction
├── ingest.py           # Unified media ingestion pipeline
├── search.py           # Query search logic
├── query_manager.py    # Interactive query interface
├── output.py           # Result rendering utilities
├── file_picker.py      # Folder selection UI with disk space estimation
├── validation.py       # Path and input validation with symlink safety
├── device.py           # Device (CPU/CUDA/MPS) management
├── cache.py            # Embedding cache with LRU eviction
├── diagnostics.py      # Health checks, stats, benchmarks
├── error_handler.py    # Error aggregation and formatting
├── index_utils.py      # Index utility functions
├── lock.py             # Process lock with TTL
├── logger.py           # Structured logging (text/JSON)
└── shutdown.py         # Graceful shutdown handling
```

### Key Design Principles

1. **Error Isolation** — one feature failure doesn't stop others
2. **Graceful Degradation** — falls back to CPU on GPU OOM
3. **Crash Recovery** — checkpoint/resume for long indexing operations
4. **Parallel Execution** — independent features run concurrently
5. **Path Safety** — symlink traversal protection via `os.path.realpath()`
6. **Retry Logic** — ChromaDB operations retry with exponential backoff

## Running Tests

```bash
pytest tests/ -v
```

Skip slow integration tests:

```bash
pytest tests/ -v -m "not slow"
```

Run with coverage:

```bash
pytest --cov=src/momento --cov-report=term-missing
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, architecture overview, and pull request guidelines.

## Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.