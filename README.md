# Momento

A semantic image search engine powered by **OpenAI CLIP** and **ChromaDB**. Search your local image library using natural language descriptions or find visually similar images — all running locally on your machine.

## Features

- **Text-to-Image Search** — Describe what you're looking for in plain English and find matching images
- **Image-to-Image Search** — Query with an image to find visually similar ones
- **Batch Ingestion** — Efficiently indexes entire directories using batched GPU/CPU processing
- **Persistent Vector Store** — Uses ChromaDB for durable, on-disk storage (no re-encoding on restart)
- **Auto Device Detection** — Automatically uses CUDA, Apple Silicon (MPS), or CPU
- **Similarity Threshold** — Filters out irrelevant results below a configurable confidence score
- **Paginated Results** — Large result sets are displayed page-by-page

## Requirements

- Python 3.12+
- `torch`, `torchvision`
- `clip` (OpenAI CLIP)
- `chromadb`
- `Pillow`

## Installation

This project uses [`uv`](https://docs.astral.sh/uv/) to manage the environment and dependencies.

```bash
# Install uv if needed
pip install uv

# Clone and sync
git clone https://github.com/your-username/momento.git
cd momento
uv sync
```

## Usage

### Quick Start

```bash
# Index a folder and start searching
uv run main.py --dir ~/Pictures

# Or just start (indexes ./images/ by default if it exists)
uv run main.py
```

### CLI Options

| Flag | Description |
|------|-------------|
| `--dir`, `-d` | Directory containing images to index on startup |
| `-h`, `--help` | Show help message |

### Interactive Menu

Once running, you'll see:

```
=== Image Search Engine ===
1. Image search (find similar images)
2. Text search (search by description)
3. Index new images from directory

Choice (1, 2, 3, or 'q' to quit):
```

- **Option 1** — Provide a path to a query image and find similar ones in your index
- **Option 2** — Type a text description (e.g., "sunset over mountains") to find matching images
- **Option 3** — Add more images from any directory without restarting

### Supported Formats

`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`

## Architecture

```
momento/
├── main.py          # CLI entry point and interactive menu
├── config.py        # Device detection, paths, constants
├── features.py      # CLIP model loading and feature extraction (single + batch)
├── index.py         # ChromaDB vector store (add, search, bulk existence check)
├── search.py        # Image and text search with similarity thresholds
├── add_images.py    # Directory scanning and batch ingestion pipeline
├── validation.py    # Input validation helpers
├── logger.py        # Centralized logging (console + file)
└── pyproject.toml   # Dependencies and project metadata
```

**Model:** OpenAI CLIP `ViT-B/16`
**Vector Store:** ChromaDB with cosine similarity (persistent on-disk)
**Logging:** Console (INFO) + file at `logs/momento.log` (DEBUG)

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
```

The similarity threshold (default `0.25`) can be adjusted in `search.py` to control result relevance.

