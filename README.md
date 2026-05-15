# clip-semantic-search

clip-semantic-search is a simple semantic search demo that uses OpenAI CLIP to compare image content with text descriptions.

## Overview

The project loads a CLIP model (`ViT-B/32`), encodes an image into a vector embedding, and compares it against a list of text queries. The best matching label is printed with its similarity score.

## Features

- Encode an image using CLIP
- Encode multiple text queries with the same CLIP model
- Compute similarity scores between image and text embeddings
- Print the best text match for a given image

## Requirements

- Python 3.12+
- `torch`
- `torchvision`
- `Pillow`
- `clip` (OpenAI CLIP)

## Installation

This project uses `uv` to manage the environment and dependencies.

1. Install `uv` if needed:

```bash
python -m pip install uv
```

2. Install dependencies from `pyproject.toml` and `uv.lock`:

```bash
uv install
```

> If you already have `uv` installed, skip step 1.

3. Run the demo using `uv`:

```bash
uv run python main.py
```

> If you have a CUDA-capable GPU, you can change the `device` variable in `main.py` from `"cpu"` to `"cuda"`.

## Usage

Run the demo script directly:

```bash
python main.py
```

The script currently loads `images/dog.jpg` and compares it against the text list:

- `dog`
- `car`
- `person running`

It then prints the best matching label and its similarity score.

## Example

```text
Best match: dog
Score: 0.XXXXXX
```

## Project Structure

- `main.py` - Example CLIP semantic search implementation
- `images/` - Sample images used by the demo
- `pyproject.toml` - Project metadata and dependency list

## Customization

To use a different image or text labels:

1. Replace the image path in `upload_images("images/dog.jpg")`
2. Update the list passed to `search_images([...], image_features)`

To use a GPU (if available), update the `device` variable:

```python
device = "cuda"
```
