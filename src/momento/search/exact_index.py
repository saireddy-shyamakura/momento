"""
exact_index.py — SQLite FTS5 exact search index for Momento V3.

Provides fast exact-match and prefix search over filenames and paths.
Used as the first-stage lookup before falling back to vector search.

Uses SQLite FTS5 for:
- Full-text search on filenames
- Prefix / suffix matching
- Exact path lookup
"""
import os
import sqlite3
import threading
from typing import List, Optional, Tuple

from ..config import CHROMA_DB_DIR
from ..logger import get_logger

logger = get_logger(__name__)

# FTS5 index lives alongside ChromaDB
_DB_PATH = os.path.join(CHROMA_DB_DIR, "exact_index.db")


class ExactIndex:
    """SQLite FTS5 index for exact filename/path search.

    Thread-safe via per-operation locks. Uses an external content
    table for reliable CRUD operations and column value storage.
    """

    def __init__(self, db_path: str = _DB_PATH):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create a thread-safe SQLite connection."""
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
        """Create content table and FTS5 virtual table."""
        conn = self._conn
        # External content table stores actual row data
        conn.execute("""
            CREATE TABLE IF NOT EXISTS file_content (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL,
                filename TEXT NOT NULL,
                ext TEXT NOT NULL DEFAULT ''
            )
        """)
        conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_file_content_path
            ON file_content(path)
        """)
        # FTS5 virtual table backed by content table
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS file_index
            USING fts5(
                path,
                filename,
                ext,
                content='file_content',
                content_rowid='rowid'
            )
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS file_index_ai AFTER INSERT ON file_content BEGIN
                INSERT INTO file_index(rowid, path, filename, ext)
                VALUES (new.rowid, new.path, new.filename, new.ext);
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS file_index_ad AFTER DELETE ON file_content BEGIN
                INSERT INTO file_index(file_index, rowid, path, filename, ext)
                VALUES ('delete', old.rowid, old.path, old.filename, old.ext);
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS file_index_au AFTER UPDATE ON file_content BEGIN
                INSERT INTO file_index(file_index, rowid, path, filename, ext)
                VALUES ('delete', old.rowid, old.path, old.filename, old.ext);
                INSERT INTO file_index(rowid, path, filename, ext)
                VALUES (new.rowid, new.path, new.filename, new.ext);
            END
        """)
        conn.commit()

    def _insert_into_content(self, conn, abs_path: str, filename: str, ext: str) -> bool:
        """Insert into content table (triggers sync to FTS5 index).

        Returns True if the path was newly inserted, False otherwise.
        """
        cursor = conn.execute(
            "INSERT OR IGNORE INTO file_content (path, filename, ext) VALUES (?, ?, ?)",
            (abs_path, filename, ext),
        )
        return cursor.rowcount > 0

    def add_paths(self, paths: List[str]) -> int:
        """Add file paths to the exact index.

        Args:
            paths: List of absolute file paths.

        Returns:
            Number of entries added.
        """
        if not paths:
            return 0

        with self._lock:
            conn = self._get_connection()
            count = 0
            for path in paths:
                abs_path = os.path.abspath(path)
                filename = os.path.basename(abs_path)
                ext = os.path.splitext(filename)[1].lower()
                try:
                    if self._insert_into_content(conn, abs_path, filename, ext):
                        count += 1
                except sqlite3.OperationalError as e:
                    logger.warning(f"Failed to index path {abs_path}: {e}")

            conn.commit()
            logger.debug(f"ExactIndex: added {count} paths")
            return count

    def search(self, query: str, top_k: int = 10) -> List[Tuple[float, str]]:
        """Search the exact index by filename or path.

        Performs an FTS5 MATCH query.  Supports:
        - Full filename match (highest score)
        - Partial filename match
        - Path prefix match

        Args:
            query: Search string (filename or path fragment).
            top_k: Maximum results.

        Returns:
            List of (score, path) tuples sorted by relevance.
        """
        if not query or not query.strip():
            return []

        with self._lock:
            conn = self._get_connection()
            q = query.strip()

            # Try exact match first via content table
            try:
                cursor = conn.execute(
                    "SELECT path FROM file_content WHERE filename = ? OR path = ? LIMIT 1",
                    (q, q),
                )
                row = cursor.fetchone()
                if row:
                    return [(1.0, row[0])]
            except sqlite3.OperationalError:
                pass

            # FTS5 match query
            fts_query = _escape_fts(q)

            try:
                cursor = conn.execute(
                    """SELECT file_content.path, file_index.rank
                       FROM file_index
                       JOIN file_content ON file_index.rowid = file_content.rowid
                       WHERE file_index MATCH ?
                       ORDER BY file_index.rank
                       LIMIT ?""",
                    (fts_query, top_k),
                )
                results = []
                for row in cursor.fetchall():
                    path, rank = row[0], row[1]
                    score = 1.0 / (1.0 + abs(rank))
                    results.append((score, path))
                return results
            except sqlite3.OperationalError:
                return self._like_search(q, top_k)

    def _like_search(self, query: str, top_k: int = 10) -> List[Tuple[float, str]]:
        """Fallback LIKE-based search when FTS5 syntax fails."""
        try:
            conn = self._get_connection()
            pattern = f"%{query}%"
            cursor = conn.execute(
                """SELECT path FROM file_content
                   WHERE filename LIKE ? OR path LIKE ?
                   LIMIT ?""",
                (pattern, pattern, top_k),
            )
            return [(0.8, row[0]) for row in cursor.fetchall()]
        except sqlite3.OperationalError:
            return []

    def remove_paths(self, paths: List[str]) -> int:
        """Remove paths from the exact index.

        Args:
            paths: List of file paths to remove.

        Returns:
            Number of entries removed.
        """
        if not paths:
            return 0

        with self._lock:
            conn = self._get_connection()
            count = 0
            for path in paths:
                abs_path = os.path.abspath(path)
                try:
                    cursor = conn.execute(
                        "DELETE FROM file_content WHERE path = ?", (abs_path,)
                    )
                    count += cursor.rowcount
                except sqlite3.OperationalError as e:
                    logger.warning(f"Failed to remove path {abs_path}: {e}")
            conn.commit()
            return count

    def clear(self) -> None:
        """Delete all entries from the exact index."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("DELETE FROM file_content")
                conn.commit()
            except sqlite3.OperationalError:
                pass

    def count(self) -> int:
        """Return the number of indexed paths."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute("SELECT COUNT(*) FROM file_content")
                return cursor.fetchone()[0]
            except sqlite3.OperationalError:
                return 0

    def close(self) -> None:
        """Close the SQLite connection."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def __len__(self) -> int:
        return self.count()


def _escape_fts(text: str) -> str:
    """Escape special characters for FTS5 MATCH queries.

    Wraps each word in quotes to avoid FTS5 syntax errors.
    """
    import re
    text = re.sub(r'[+*"()~<>@]', ' ', text)
    words = text.split()
    if not words:
        return '""'
    if len(words) == 1:
        return f'"{words[0]}"'
    return ' '.join(f'"{w}"' for w in words)