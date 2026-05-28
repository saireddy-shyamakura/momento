# Contributing to Momento

Thank you for your interest in contributing to Momento! This document provides guidelines and instructions for contributing.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Running Tests](#running-tests)
5. [Architecture Overview](#architecture-overview)
6. [Pull Request Guidelines](#pull-request-guidelines)
7. [Coding Standards](#coding-standards)

## Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/momento.git`
3. Create a branch: `git checkout -b feature/my-feature`

## Development Setup

### Prerequisites

- Python 3.12 or later
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Install in Development Mode

Using uv (recommended):
```bash
cd momento
uv sync
```

Using pip:
```bash
cd momento
pip install -e ".[dev]"
```

### Install Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

## Running Tests

### Run All Tests

```bash
pytest
```

### Run Unit Tests Only (Skip Slow Integration Tests)

```bash
pytest -m "not slow"
```

### Run Tests with Coverage

```bash
pytest --cov=src/momento --cov-report=term-missing
```

### Run Specific Test File

```bash
pytest tests/test_search.py -v
```

## Architecture Overview

Momento follows a modular architecture organized around:

```
src/momento/
├── __init__.py        # Package metadata and version
├── app_controller.py  # Main application orchestrator
├── cli.py             # Command-line interface
├── config.py          # Central configuration
├── index.py           # ChromaDB vector store wrapper
├── indexer.py         # Indexing orchestrator (parallel execution)
├── features.py        # CLIP model and feature extraction
├── add_images.py      # Image ingestion pipeline
├── augment.py         # Image augmentation transforms
├── video.py           # Video keyframe extraction
├── yolo.py            # YOLO object detection
├── ocr.py             # OCR text extraction
├── ingest.py          # Unified media ingestion pipeline
├── search.py          # Query search logic
├── query_manager.py   # Interactive query interface
├── output.py          # Result rendering
├── file_picker.py     # Folder selection UI
├── validation.py      # Path and input validation
├── device.py          # Device (CPU/CUDA/MPS) management
├── cache.py           # Embedding cache with LRU eviction
├── diagnostics.py     # Health checks, stats, benchmarks
├── error_handler.py   # Error aggregation
├── index_utils.py     # Index utility functions
├── lock.py            # Process lock with TTL
├── logger.py          # Structured logging
└── shutdown.py        # Graceful shutdown handling
```

### Key Design Principles

1. **Error Isolation**: One feature failure doesn't stop others
2. **Graceful Degradation**: Falls back to CPU on GPU OOM
3. **Crash Recovery**: Checkpoint/resume for long indexing operations
4. **Parallel Execution**: Independent features run concurrently
5. **Path Safety**: Symlink traversal protection via `os.path.realpath()`

## Pull Request Guidelines

1. **One feature per PR**: Keep changes focused and reviewable
2. **Write tests**: Include unit tests for new functionality
3. **Update docs**: Update relevant docstrings and documentation
4. **Pass CI**: Ensure all tests pass and linting is clean
5. **Type hints**: Add type hints to all public functions
6. **Changelog**: Add an entry to the changelog if one exists

### PR Checklist

Before submitting your PR:

- [ ] Code follows project coding standards
- [ ] Tests pass: `pytest`
- [ ] Linting clean: `ruff check .`
- [ ] Type checks: `mypy src/momento/`
- [ ] Documentation updated
- [ ] No new warnings or errors

## Coding Standards

### Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use [ruff](https://github.com/astral-sh/ruff) for linting
- Use [mypy](https://mypy-lang.org/) for type checking

### Docstrings

Use Google-style docstrings:

```python
def function_name(param1: str, param2: int) -> bool:
    """Short description.

    Longer description if needed.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.

    Raises:
        ValueError: If param1 is invalid.
    """
```

### Imports

Order imports in groups:
1. Standard library
2. Third-party libraries
3. Local imports

### Error Handling

- Use specific exceptions (not bare `except:`)
- Log errors with appropriate context
- Let fatal errors propagate to the top-level handler

## Issue Templates

Please use the appropriate issue template:

- **Bug Report**: For reporting bugs
- **Feature Request**: For suggesting enhancements
- **Question**: For questions about usage