# Momento

A local, multi-modal semantic search engine built on **OpenAI CLIP** and **ChromaDB**. Momento enables powerful image and video search entirely on your machine, with no cloud dependencies.

## Features

### Core Capabilities
- **Image-to-Image Search** — find visually similar images from a query image
- **Text-to-Image Search** — describe a scene and find matching media
- **Video Keyframe Indexing** — extract and index frames from video files _(optional)_
- **YOLO Object Detection** — detect and embed individual object crops for fine-grained search _(optional)_
- **OCR Text Extraction** — extract on-image text and index it for text-based search _(optional)_

### Smart Indexing
- **Multi-Embedding Augmentation** — index 5 augmented views per image (flip, crop, brightness, contrast, rotate) for better recall _(optional)_
- **Score Aggregation** — multi-vector results (augmentations, YOLO crops, OCR) aggregated by maximum score per source file for cleaner, deduplicated results
- **Parallel Execution** — Videos, Objects, and OCR features run concurrently for 2-4x faster indexing
- **Automatic Device Selection** — CUDA (NVIDIA), MPS (Apple Silicon), or CPU with OOM fallback
- **Embedding Cache** — LRU-evicted cache with mtime-based invalidation to speed up re-indexing
- **Disk Space Estimation** — previews estimated vector count and storage needs, checks free disk space before indexing
- **Memory Pre-check** — warns if less than 2 GB free system memory before starting indexing

### Reliability & Persistence
- **Crash Recovery** — checkpoint/resume so interrupted indexing picks up where it left off
- **Graceful Shutdown** — SIGINT/SIGTERM handlers complete the current batch before exiting
- **Process Lock** — PID-based lock file prevents multiple instances from running simultaneously, with 6-hour TTL and automatic stale-lock cleanup
- **Persistent Storage** — all data persists in ChromaDB with automatic backups
- **Storage Management** — cache clearing, database optimization, backup/restore
- **Auto-Repair** — automatically repairs corrupted ChromaDB on startup; falls back to a clear error with `--reset` guidance if repair fails
- **Version Mismatch Detection** — stores ChromaDB and Momento versions in collection metadata, warns on version mismatches to prevent silent data corruption

### Customization
- **Model Selection** — choose from 5 CLIP models (ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14@336px, ConvNeXt-B)
- **Feature Toggles** — enable/disable any feature (multi-embed, video, YOLO, OCR) independently
- **Flexible Configuration** — TOML files, environment variables, or CLI flags

### Developer Features
- **Structured Logging** — JSON or text format with timing and event tracking
- **Health Diagnostics** — system checks, index statistics, performance benchmarks
- **Programmatic API** — full Python API for integration

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

## Customization

### Model Selection

Momento supports 5 CLIP models with different speed/quality tradeoffs:

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| ViT-B/32 | 338 MB | ★★★★★ | ★★★ | Fast indexing, limited resources |
| ViT-B/16 | 355 MB | ★★★★ | ★★★★ | **Default - balanced** |
| ViT-L/14 | 927 MB | ★★★ | ★★★★★ | High-quality search |
| ViT-L/14@336px | 927 MB | ★★ | ★★★★★ | Maximum quality |
| ConvNeXt-B | 400 MB | ★★★★ | ★★★★ | Modern architecture |

Select a model via:
- **CLI**: `momento --model ViT-L/14 --dir ~/Pictures`
- **Environment**: `export MOMENTO_MODEL_NAME=ViT-L/14`
- **Config file**: Edit `~/.config/momento/config.toml`

### Feature Management

All features can be toggled independently:

| Feature | CLI Flag | Env Variable | Enabled by Default |
|---------|----------|--------------|------------------|
| Multi-Embedding Augmentation | `--no-multi-embed` | `MOMENTO_ENABLE_MULTI_EMBED` | ✅ Yes |
| Video Keyframe Indexing | `--no-video` | `MOMENTO_ENABLE_VIDEO_INDEXING` | ✅ Yes |
| YOLO Object Detection | `--no-yolo` | `MOMENTO_ENABLE_YOLO` | ✅ Yes |
| OCR Text Extraction | `--no-ocr` | `MOMENTO_ENABLE_OCR` | ✅ Yes |

**Examples:**
```bash
# Minimal indexing (fast, low resource)
momento --no-multi-embed --no-video --no-yolo --no-ocr --dir ~/pictures

# Document-focused (OCR + basic search)
momento --no-multi-embed --no-video --no-yolo --dir ~/documents

# High-quality search (all features enabled)
momento --model ViT-L/14 --dir ~/pictures
```

### Configuration

Momento loads configuration in this priority order:
1. **CLI flags** (highest priority)
2. **Environment variables** (`MOMENTO_*` prefix)
3. **Config file** (`~/.config/momento/config.toml`)
4. **Defaults** (lowest priority)

**Example config file** (`~/.config/momento/config.toml`):
```toml
[embedding]
model_name = "ViT-L/14"

[features]
enable_multi_embed = true
enable_video_indexing = true
enable_yolo = true
enable_ocr = true

[similarity]
similarity_threshold = 0.20
max_search_results = 50

[indexing]
indexing_batch_size = 32
progress_bar_enabled = true

[video]
video_frame_interval = 2.0
max_frames_per_video = 50

[yolo]
yolo_model = "yolov8n.pt"
yolo_confidence_threshold = 0.35

[ocr]
ocr_languages = ["en"]
ocr_min_text_length = 3

[augmentation]
augmentation_count = 5

[cache]
cache_max_size_gb = 10

[logging]
log_format = "text"
log_level = "INFO"
```

### Persistent Storage

All indexed data is stored persistently:
- **Database**: `~/.local/share/momento/chroma_db/chroma.sqlite3`
- **Cache**: `~/.local/share/momento/embedding_cache/`
- **Logs**: `~/.local/share/momento/logs/`
- **Checkpoints**: `~/.local/share/momento/indexing_checkpoint.json`

See [PERSISTENT_STORAGE.md](PERSISTENT_STORAGE.md) for backup, restore, and management procedures.

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
| `--model MODEL` | CLIP model: `ViT-B/32`, `ViT-B/16` (default), `ViT-L/14`, `ViT-L/14@336px`, `ConvNeXt-B` |
| `--no-multi-embed` | Disable multi-embedding augmentation |
| `--no-video` | Disable video keyframe indexing |
| `--no-yolo` | Disable YOLO object detection |
| `--no-ocr` | Disable OCR text extraction |
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

#### `momento storage`

Manage storage, cache, and backups:

| Action | Command | Description |
|--------|---------|-------------|
| **Info** | `momento storage info` | Show storage usage breakdown |
| **Clear Cache** | `momento storage clear-cache` | Free disk space by clearing embedding cache |
| **Clear Logs** | `momento storage clear-logs` | Remove old log files |
| **Optimize** | `momento storage optimize` | Optimize database (VACUUM + ANALYZE) |
| **Backup** | `momento storage backup` | Create timestamped database backup |
| **Restore** | `momento storage restore <backup.db>` | Restore database from backup file |
| **Export** | `momento storage export -o backup.sql` | Export database to SQL dump |

Examples:
```bash
# View storage usage
momento storage info

# Create backup before re-indexing
momento storage backup

# Restore from backup if needed
momento storage restore ~/backups/momento/chroma_backup_20250530_120000.db

# Optimize database after heavy indexing
momento storage optimize
```

## Supported File Formats

- **Images**: `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`
- **Videos**: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.flv`

## Crash Recovery & Checkpoints

Momento automatically saves checkpoints during indexing, allowing recovery from interruptions:

```bash
# Start indexing (Ctrl+C to interrupt)
momento --dir ~/Pictures

# Checkpoint automatically saved
# Run again to resume from last position
momento --dir ~/Pictures
```

**How it works:**
1. Checkpoint created when indexing starts
2. Progress tracked per feature (images, video, YOLO, OCR)
3. On interrupt, checkpoint saved with current state
4. On resume, completed features skipped, in-progress feature resumed
5. After successful completion, checkpoint cleared

Checkpoint location: `~/.local/share/momento/indexing_checkpoint.json`

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

### Composite ID System

Momento uses a composite ID system to track multiple vectors belonging to the same source file. When a single image produces multiple embeddings (original + augmentations + YOLO object crops + OCR text), each vector is stored with a composite ID like:

```
path/to/photo.jpg|||orig       # Original image embedding
path/to/photo.jpg|||flip       # Horizontal flip augmentation
path/to/photo.jpg|||yolo_cat   # YOLO-detected cat crop
path/to/photo.jpg|||ocr        # OCR text embedding
```

This enables **score aggregation** during search — results from the same source file are grouped and ranked by their maximum score, producing clean, deduplicated results.

### Key Design Principles

1. **Error Isolation** — one feature failure doesn't stop others
2. **Graceful Degradation** — falls back to CPU on GPU OOM
3. **Crash Recovery** — checkpoint/resume for long indexing operations
4. **Score Aggregation** — multi-vector results grouped and ranked by max score per source file
5. **Process Locking** — single-instance guard with stale-lock cleanup
6. **Auto-Repair** — corrupted ChromaDB automatically recreated on startup
7. **Parallel Execution** — independent features run concurrently
8. **Path Safety** — symlink traversal protection via `os.path.realpath()`
9. **Retry Logic** — ChromaDB operations retry with exponential backoff
10. **Disk & Memory Checks** — pre-indexing validation prevents out-of-space/failures
11. **Version Mismatch Detection** — schema version tracking prevents silent data corruption

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