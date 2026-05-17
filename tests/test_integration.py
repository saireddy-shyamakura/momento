import os
import shutil
import pytest

from momento.index import Index
from momento.add_images import add_images
from momento.search import text_search
from momento.config import SIMILARITY_THRESHOLD

pytestmark = pytest.mark.slow

@pytest.fixture
def integration_images(tmp_path):
    """Copies a small subset of real images into a tmp_path directory for testing."""
    test_images_dir = tmp_path / "images"
    test_images_dir.mkdir()
    
    # Base images folder
    # In test environment, the base is the project root, so "images"
    project_images_dir = os.path.join(os.path.dirname(__file__), "..", "images")
    if not os.path.exists(project_images_dir):
        pytest.skip(f"Could not find images directory at {project_images_dir}")
        
    # Copy a small subset (up to 3 images)
    copied = 0
    for file in os.listdir(project_images_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            src = os.path.join(project_images_dir, file)
            dst = os.path.join(str(test_images_dir), file)
            shutil.copy2(src, dst)
            copied += 1
            if copied >= 3:
                break
                
    if copied == 0:
        pytest.skip("No images found to use for integration test.")
        
    return str(test_images_dir)

@pytest.fixture
def integration_index(tmp_path):
    """Provides a fresh Index for integration testing."""
    db_path = str(tmp_path / "chroma_db")
    return Index(db_path=db_path)

def test_integration_add_and_search(integration_images, integration_index):
    """Test: add_images() -> text_search() returns at least one result with score >= threshold."""
    # Add images
    num_added = add_images(integration_images, integration_index)
    assert num_added > 0
    assert integration_index.get_vector_count() == num_added
    
    results = text_search("a picture", integration_index, top_k=1, threshold=SIMILARITY_THRESHOLD)
    
    assert len(results) > 0
    assert results[0][0] >= SIMILARITY_THRESHOLD

def test_integration_add_images_idempotency(integration_images, integration_index):
    """Test: calling add_images() twice on the same folder leaves get_vector_count() unchanged."""
    num_added_first = add_images(integration_images, integration_index)
    assert num_added_first > 0
    initial_count = integration_index.get_vector_count()
    
    # Second addition should return 0 (no new images)
    num_added_second = add_images(integration_images, integration_index)
    assert num_added_second == 0
    assert integration_index.get_vector_count() == initial_count

def test_integration_search_empty_index(integration_index):
    """Test: text_search() on a fresh empty index returns [] without exception."""
    results = text_search("anything", integration_index, top_k=3, threshold=0.0)
    assert results == []
