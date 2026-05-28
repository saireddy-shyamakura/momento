"""
Create test fixture files for Phase 1 edge-case testing.

Creates:
- corrupt.jpg (truncated JPEG header)
- zero_byte.png (empty file)
- huge.txt (valid extension but text content)
- nested/deep/path/image.webp
- image_with_unicode_🎉.jpg
"""

import os

FIXTURE_DIR = os.path.dirname(os.path.abspath(__file__))


def create_fixtures():
    """Create all edge-case test fixtures."""

    # 1. corrupt.jpg — truncated JPEG header (missing data)
    corrupt_jpg = os.path.join(FIXTURE_DIR, "corrupt.jpg")
    with open(corrupt_jpg, "wb") as f:
        f.write(b"\xff\xd8\xff")  # Just the SOI marker, no image data

    # 2. zero_byte.png — completely empty file
    zero_png = os.path.join(FIXTURE_DIR, "zero_byte.png")
    with open(zero_png, "wb") as f:
        pass  # Empty file

    # 3. huge.txt — text file with image-like extension name but .txt content
    huge_txt = os.path.join(FIXTURE_DIR, "huge.txt")
    with open(huge_txt, "w") as f:
        f.write("This is not an image file. It just has a .txt extension.\n")

    # 4. nested/deep/path/image.webp
    nested_dir = os.path.join(FIXTURE_DIR, "nested", "deep", "path")
    os.makedirs(nested_dir, exist_ok=True)
    webp_file = os.path.join(nested_dir, "image.webp")
    # Minimal valid webp file header
    with open(webp_file, "wb") as f:
        f.write(b"RIFF" + b"\x00" * 8 + b"WEBP")

    # 5. image_with_unicode_🎉.jpg — file with unicode in name
    unicode_file = os.path.join(FIXTURE_DIR, "image_with_unicode_🎉.jpg")
    with open(unicode_file, "wb") as f:
        f.write(b"\xff\xd8\xff\xe0" + b"\x00" * 16)

    print(f"Created fixture files in {FIXTURE_DIR}")
    print("  - corrupt.jpg")
    print("  - zero_byte.png")
    print("  - huge.txt")
    print("  - nested/deep/path/image.webp")
    print("  - image_with_unicode_🎉.jpg")


if __name__ == "__main__":
    create_fixtures()