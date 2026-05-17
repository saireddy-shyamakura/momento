# Momento

A semantic image search engine powered by **OpenAI CLIP** and **ChromaDB**. Search your local image library using natural language descriptions or find visually similar images — all running locally on your machine.

## Features

- **Text-to-Image Search** — Describe what you're looking for in plain English and find matching images
- **Image-to-Image Search** — Query with an image to find visually similar ones
- **Batch Ingestion** — Efficiently indexes entire directories using batched GPU/CPU processing
- **Persistent Vector Store** — Uses ChromaDB for durable, on-disk storage (no re-encoding on restart)
- **Auto Device Detection** — Automatically uses CUDA, Apple Silicon (MPS), or CPU
- **Similarity Threshold** — Filters out irrelevant results below a configurable confidence score
- **Progress Bar** — Visual progress during batch indexing (requires `tqdm`)
- **Paginated Results** — Large result sets are displayed page-by-page
- **Concurrency Lock** — Prevents multiple instances from corrupting the index simultaneously

## Requirements

- Python 3.12+
- `torch`, `torchvision`
- `clip` (OpenAI CLIP)
- `chromadb`
- `Pillow`
- `platformdirs`

Optional:
- `tqdm` — for progress bars during indexing (`uv sync --extra progress`)

## Installation

This project uses [`uv`](https://docs.astral.sh/uv/) to manage the environment and dependencies.

```bash
# Install uv if needed
pip install uv

# Clone and sync
git clone https://github.com/your-username/momento.git
cd momento
uv sync

# With progress bar support
uv sync --extra progress
```

## Usage

### Quick Start

```bash
# Index a folder and start searching
uv run momento --dir ~/Pictures

# Or just start the interactive menu
uv run momento
```

### CLI Flags

| Flag | Description |
|------|-------------|
| `--dir`, `-d PATH` | Directory containing images to index on startup |
| `--version` | Print version and exit |
| `--reset` | Delete all indexed entries and exit |
| `--count` | Print number of currently indexed images and exit |
| `--verify` | Scan the index and remove entries whose files no longer exist on disk |
| `--threshold FLOAT` | Override the similarity threshold for this session (0.0–1.0, default 0.20) |
| `--open` | After each search, automatically open the top result in the system image viewer |
| `-h`, `--help` | Show help message and exit |

### Interactive Menu

Once running, you'll see:

```
=== Image Search Engine ===
1. Image search (find similar images)
2. Text search (search by description)
3. Index new images from directory

Choice (1, 2, 3, or 'q' to quit):
```

- **Option 1** — Provide a path to a query image and find visually similar ones in your index
- **Option 2** — Type a text description (e.g., "sunset over mountains") to find matching images
- **Option 3** — Add more images from any directory without restarting

### Supported Formats

`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`

## Architecture

```
momento/
├── pyproject.toml               # Dependencies, entry point, build config
├── src/
│   └── momento/
│       ├── __init__.py          # Package version
│       ├── cli.py               # Entry point: argument parsing, interactive menu, lock acquire/release
│       ├── lock.py              # LockFile: prevents concurrent processes
│       ├── config.py            # Device detection, paths, constants (uses platformdirs)
│       ├── features.py          # CLIP model loading and feature extraction (single + batch)
│       ├── index.py             # ChromaDB vector store (add, search, bulk existence check)
│       ├── search.py            # Image and text search with similarity thresholds
│       ├── add_images.py        # Directory scanning and batch ingestion pipeline
│       ├── validation.py        # Input validation helpers
│       ├── output.py            # Result rendering: format_bar(), render_result(), open_file()
│       └── logger.py            # Centralized logging (console + rotating file)
└── tests/
    ├── test_add_images.py
    ├── test_config.py
    ├── test_index.py
    ├── test_integration.py      # End-to-end pipeline tests (marked @pytest.mark.slow)
    ├── test_output.py
    ├── test_search.py
    └── test_validation.py
```

**Model:** OpenAI CLIP `ViT-B/16`  
**Vector Store:** ChromaDB with cosine similarity (persistent on-disk)  
**Data Directory:** `~/.local/share/momento/` (Linux/macOS) / `%APPDATA%\momento\` (Windows)  
**Logging:** Console (INFO) + rotating file at `~/.local/share/momento/logs/momento.log` (DEBUG)

## Configuration

Device detection is fully automatic in `config.py`:

| Hardware | Device |
|----------|--------|
| NVIDIA GPU | `cuda` |
| Apple Silicon | `mps` |
| Everything else | `cpu` |

Key constants in `config.py`:

```python
MODEL_NAME = "ViT-B/16"
SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
SIMILARITY_THRESHOLD = 0.20
```

The similarity threshold can be overridden per-session with `--threshold`:

```bash
uv run momento --threshold 0.30
```

## Running Tests

```bash
# Fast unit + property tests (skip slow integration tests)
uv run pytest tests/ -v -m "not slow"

# Full suite including integration tests (requires real images in ./images/)
uv run pytest tests/ -v
```
