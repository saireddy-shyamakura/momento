"""
metadata_store.py — SQLite-backed metadata store for Momento V3.

Stores file metadata separately from the vector index to enable:
- Fast metadata filtering (file type, date, size)
- Filtering before or after vector search
- Independence from ChromaDB schema

Schema:
- path: TEXT PRIMARY KEY
- filename: TEXT
- ext: TEXT (file extension)
- file_size: INTEGER (bytes)
- indexed_at: TEXT (ISO 8601 timestamp)
- ocr_text: TEXT (optional, extracted text)
- objects: TEXT (optional, JSON list of detected objects)
- custom: TEXT (optional, JSON dict for extensibility)
"""
import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from ..config import CHROMA_DB_DIR
from ..logger import get_logger

logger = get_logger(__name__)

_DB_PATH = os.path.join(CHROMA_DB_DIR, "metadata.db")


class MetadataStore:
    """Thread-safe SQLite store for file metadata.

    Designed to complement ChromaDB — keeps searchable metadata
    separate from vector data.
    """

    def __init__(self, db_path: str = _DB_PATH):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None

    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
            self._conn = sqlite3.connect(
                self._db_path,
                check_same_thread=False,
            )
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=OFF")
            self._init_schema()
        return self._conn

    def _init_schema(self) -> None:
        conn = self._conn
        conn.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                path TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                ext TEXT NOT NULL DEFAULT '',
                file_size INTEGER NOT NULL DEFAULT 0,
                indexed_at TEXT NOT NULL,
                ocr_text TEXT DEFAULT '',
                objects TEXT DEFAULT '[]',
                custom TEXT DEFAULT '{}'
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_metadata_ext
            ON metadata(ext)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_metadata_indexed_at
            ON metadata(indexed_at)
        """)
        conn.commit()

    def add_file(self, path: str, file_size: int = 0,
                 ocr_text: str = "",
                 objects: Optional[List[str]] = None,
                 custom: Optional[Dict[str, Any]] = None) -> None:
        """Add or update metadata for a file.

        Args:
            path: Absolute file path.
            file_size: File size in bytes.
            ocr_text: Extracted OCR text (if any).
            objects: List of detected object labels.
            custom: Arbitrary custom metadata dict.
        """
        abs_path = os.path.abspath(path)
        filename = os.path.basename(abs_path)
        ext = os.path.splitext(filename)[1].lower()
        now = datetime.now(timezone.utc).isoformat()

        with self._lock:
            conn = self._get_connection()
            conn.execute("""
                INSERT OR REPLACE INTO metadata
                    (path, filename, ext, file_size, indexed_at,
                     ocr_text, objects, custom)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                abs_path, filename, ext, file_size, now,
                ocr_text,
                json.dumps(objects or []),
                json.dumps(custom or {}),
            ))
            conn.commit()

    def batch_add(self, entries: List[Dict[str, Any]]) -> int:
        """Add multiple file metadata entries in a transaction.

        Args:
            entries: List of dicts with keys:
                - path (required)
                - file_size (optional)
                - ocr_text (optional)
                - objects (optional)
                - custom (optional)

        Returns:
            Number of entries added.
        """
        if not entries:
            return 0

        with self._lock:
            conn = self._get_connection()
            now = datetime.now(timezone.utc).isoformat()
            count = 0
            for entry in entries:
                path = entry.get("path", "")
                if not path:
                    continue
                abs_path = os.path.abspath(path)
                filename = os.path.basename(abs_path)
                ext = os.path.splitext(filename)[1].lower()
                conn.execute("""
                    INSERT OR REPLACE INTO metadata
                        (path, filename, ext, file_size, indexed_at,
                         ocr_text, objects, custom)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    abs_path, filename, ext,
                    entry.get("file_size", 0), now,
                    entry.get("ocr_text", ""),
                    json.dumps(entry.get("objects", [])),
                    json.dumps(entry.get("custom", {})),
                ))
                count += 1
            conn.commit()
            return count

    def get_file(self, path: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific file.

        Args:
            path: File path.

        Returns:
            Dict of metadata, or None if not found.
        """
        abs_path = os.path.abspath(path)
        with self._lock:
            conn = self._get_connection()
            cursor = conn.execute(
                "SELECT * FROM metadata WHERE path = ?", (abs_path,)
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return self._row_to_dict(row)

    def filter(self, ext: Optional[str] = None,
               min_size: Optional[int] = None,
               max_size: Optional[int] = None,
               after: Optional[str] = None,
               before: Optional[str] = None,
               ocr_contains: Optional[str] = None,
               limit: int = 100) -> List[Dict[str, Any]]:
        """Filter files by metadata criteria.

        Args:
            ext: Filter by file extension (e.g. '.jpg').
            min_size: Minimum file size in bytes.
            max_size: Maximum file size in bytes.
            after: ISO timestamp — only files indexed after this.
            before: ISO timestamp — only files indexed before this.
            ocr_contains: Filter by OCR text containing this string.
            limit: Maximum results.

        Returns:
            List of metadata dicts matching the criteria.
        """
        conditions: List[str] = []
        params: List[Any] = []

        if ext:
            conditions.append("ext = ?")
            params.append(ext.lower())
        if min_size is not None:
            conditions.append("file_size >= ?")
            params.append(min_size)
        if max_size is not None:
            conditions.append("file_size <= ?")
            params.append(max_size)
        if after:
            conditions.append("indexed_at >= ?")
            params.append(after)
        if before:
            conditions.append("indexed_at <= ?")
            params.append(before)
        if ocr_contains:
            conditions.append("ocr_text LIKE ?")
            params.append(f"%{ocr_contains}%")

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        with self._lock:
            conn = self._get_connection()
            cursor = conn.execute(
                f"SELECT * FROM metadata WHERE {where_clause} ORDER BY indexed_at DESC LIMIT ?",
                (*params, limit),
            )
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    def get_all_paths(self) -> List[str]:
        """Get all indexed file paths."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.execute("SELECT path FROM metadata ORDER BY path")
            return [row[0] for row in cursor.fetchall()]

    def count(self) -> int:
        """Return the number of metadata entries."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.execute("SELECT COUNT(*) FROM metadata")
            return cursor.fetchone()[0]

    def remove_path(self, path: str) -> bool:
        """Remove metadata for a file.

        Returns:
            True if an entry was removed.
        """
        abs_path = os.path.abspath(path)
        with self._lock:
            conn = self._get_connection()
            cursor = conn.execute("DELETE FROM metadata WHERE path = ?", (abs_path,))
            conn.commit()
            return cursor.rowcount > 0

    def clear(self) -> None:
        """Delete all metadata entries."""
        with self._lock:
            conn = self._get_connection()
            conn.execute("DELETE FROM metadata")
            conn.commit()

    def close(self) -> None:
        """Close the SQLite connection."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    @staticmethod
    def _row_to_dict(row) -> Dict[str, Any]:
        """Convert a sqlite3.Row to a dict with parsed JSON fields."""
        return {
            "path": row[0],
            "filename": row[1],
            "ext": row[2],
            "file_size": row[3],
            "indexed_at": row[4],
            "ocr_text": row[5],
            "objects": json.loads(row[6]) if row[6] else [],
            "custom": json.loads(row[7]) if row[7] else {},
        }