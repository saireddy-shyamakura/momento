"""
storage_manager.py — Storage management and persistence operations.

Handles:
- Cache management (info, clear, optimization)
- Database maintenance (vacuum, check, repair)
- Backup and restore operations
- Storage statistics and reporting
"""

import os
import json
import shutil
import re
import sqlite3
from typing import Dict, Tuple, Optional, List
from datetime import datetime

from .config import (
    CHROMA_DB_DIR,
    EMBEDDING_CACHE_DIR,
    LOG_DIR,
    BASE_DIR,
    CACHE_MAX_SIZE_GB,
)
from .logger import get_logger

logger = get_logger(__name__)

# Only these SQL statement types are allowed during database import
_ALLOWED_SQL_PREFIXES = (
    "INSERT", "CREATE", "BEGIN", "COMMIT", "ROLLBACK",
    "PRAGMA", "UPDATE", "DELETE", "REPLACE",
)


class StorageManager:
    """Manage Momento's persistent storage."""
    
    @staticmethod
    def get_storage_usage() -> Dict[str, int]:
        """Get storage usage breakdown in bytes.
        
        Returns:
            Dict with keys: total, database, cache, logs, other
        """
        usage = {
            "total": 0,
            "database": 0,
            "cache": 0,
            "logs": 0,
            "other": 0,
        }
        
        # Database size
        db_file = os.path.join(CHROMA_DB_DIR, "chroma.sqlite3")
        if os.path.exists(db_file):
            usage["database"] = os.path.getsize(db_file)
        
        # Cache size
        if os.path.exists(EMBEDDING_CACHE_DIR):
            usage["cache"] = _get_dir_size(EMBEDDING_CACHE_DIR)
        
        # Logs size
        if os.path.exists(LOG_DIR):
            usage["logs"] = _get_dir_size(LOG_DIR)
        
        # Other (metadata, checkpoints, etc.)
        usage["other"] = _get_dir_size(BASE_DIR) - sum(v for k, v in usage.items() if k != "total")
        
        # Total
        usage["total"] = sum(v for k, v in usage.items() if k != "total")
        
        return usage
    
    @staticmethod
    def format_size(bytes: int) -> str:
        """Format bytes as human-readable string."""
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if bytes < 1024:
                return f"{bytes:.2f} {unit}"
            bytes /= 1024
        return f"{bytes:.2f} PB"
    
    @staticmethod
    def clear_cache() -> int:
        """Clear embedding cache.
        
        Returns:
            Number of files deleted
        """
        if not os.path.exists(EMBEDDING_CACHE_DIR):
            return 0
        
        deleted = 0
        try:
            for fname in os.listdir(EMBEDDING_CACHE_DIR):
                if fname.endswith((".npz", ".pkl", ".access")):
                    fpath = os.path.join(EMBEDDING_CACHE_DIR, fname)
                    try:
                        os.remove(fpath)
                        deleted += 1
                    except OSError as e:
                        logger.warning(f"Failed to delete cache file {fpath}: {e}")
            
            logger.info(f"Cleared {deleted} cache files")
        except OSError as e:
            logger.error(f"Failed to clear cache: {e}")
        
        return deleted
    
    @staticmethod
    def clear_logs() -> int:
        """Clear application logs.
        
        Returns:
            Number of files deleted
        """
        if not os.path.exists(LOG_DIR):
            return 0
        
        deleted = 0
        try:
            for fname in os.listdir(LOG_DIR):
                if fname.endswith(".log"):
                    fpath = os.path.join(LOG_DIR, fname)
                    try:
                        os.remove(fpath)
                        deleted += 1
                    except OSError as e:
                        logger.warning(f"Failed to delete log file {fpath}: {e}")
            
            logger.info(f"Cleared {deleted} log files")
        except OSError as e:
            logger.error(f"Failed to clear logs: {e}")
        
        return deleted
    
    @staticmethod
    def optimize_database() -> bool:
        """Optimize ChromaDB database (VACUUM + ANALYZE).
        
        Returns:
            True if successful
        """
        db_file = os.path.join(CHROMA_DB_DIR, "chroma.sqlite3")
        if not os.path.exists(db_file):
            logger.warning("Database file not found")
            return False
        
        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            
            # Run optimization commands
            cursor.execute("VACUUM;")
            cursor.execute("ANALYZE;")
            cursor.execute("PRAGMA optimize;")
            
            conn.commit()
            conn.close()
            
            logger.info("Database optimized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to optimize database: {e}")
            return False
    
    @staticmethod
    def get_database_stats() -> Dict[str, any]:
        """Get statistics about the database.
        
        Returns:
            Dict with count, size, etc.
        """
        db_file = os.path.join(CHROMA_DB_DIR, "chroma.sqlite3")
        if not os.path.exists(db_file):
            return {"error": "Database not found"}
        
        stats = {
            "size_bytes": os.path.getsize(db_file),
            "size_formatted": StorageManager.format_size(os.path.getsize(db_file)),
            "embeddings_count": 0,
            "collections_count": 0,
        }
        
        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            
            # Count embeddings
            cursor.execute("SELECT COUNT(*) FROM embeddings;")
            result = cursor.fetchone()
            if result:
                stats["embeddings_count"] = result[0]
            
            # Count collections
            cursor.execute("""
                SELECT COUNT(DISTINCT collection_id) FROM embeddings;
            """)
            result = cursor.fetchone()
            if result:
                stats["collections_count"] = result[0]
            
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to get database stats: {e}")
            stats["error"] = str(e)
        
        return stats
    
    @staticmethod
    def backup_database(backup_dir: Optional[str] = None) -> Optional[str]:
        """Create backup of database.
        
        Args:
            backup_dir: Directory to store backup. Defaults to ~/backups/momento
            
        Returns:
            Path to backup file, or None if failed
        """
        if backup_dir is None:
            backup_dir = os.path.expanduser("~/backups/momento")
        
        db_file = os.path.join(CHROMA_DB_DIR, "chroma.sqlite3")
        if not os.path.exists(db_file):
            logger.error("Database file not found")
            return None
        
        try:
            os.makedirs(backup_dir, exist_ok=True)
            
            # Create timestamped backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = os.path.join(backup_dir, f"chroma_backup_{timestamp}.db")
            
            shutil.copy2(db_file, backup_file)
            logger.info(f"Database backed up to: {backup_file}")
            
            return backup_file
        except Exception as e:
            logger.error(f"Failed to backup database: {e}")
            return None
    
    @staticmethod
    def restore_database(backup_file: str) -> bool:
        """Restore database from backup file.
        
        Args:
            backup_file: Path to backup file
            
        Returns:
            True if successful
        """
        if not os.path.exists(backup_file):
            logger.error(f"Backup file not found: {backup_file}")
            return False
        
        db_file = os.path.join(CHROMA_DB_DIR, "chroma.sqlite3")
        
        try:
            os.makedirs(CHROMA_DB_DIR, exist_ok=True)
            shutil.copy2(backup_file, db_file)
            logger.info(f"Database restored from: {backup_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to restore database: {e}")
            return False
    
    @staticmethod
    def export_database(export_file: str) -> bool:
        """Export database to SQL dump file.
        
        Args:
            export_file: Path to write SQL dump
            
        Returns:
            True if successful
        """
        db_file = os.path.join(CHROMA_DB_DIR, "chroma.sqlite3")
        if not os.path.exists(db_file):
            logger.error("Database file not found")
            return False
        
        try:
            conn = sqlite3.connect(db_file)
            
            with open(export_file, 'w') as f:
                for line in conn.iterdump():
                    f.write(f"{line}\n")
            
            conn.close()
            logger.info(f"Database exported to: {export_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to export database: {e}")
            return False
    
    @staticmethod
    def _validate_sql_statement(statement: str) -> bool:
        """Check that a SQL statement only uses allowed command types.

        This prevents dangerous SQL such as DROP TABLE, ATTACH DATABASE,
        or shell commands from being executed during import.

        Args:
            statement: Single SQL statement (non-empty, stripped).

        Returns:
            True if the statement is allowed.
        """
        first_word = statement.split(None, 1)[0].upper()
        return first_word in _ALLOWED_SQL_PREFIXES

    @staticmethod
    def import_database(import_file: str) -> bool:
        """Import database from SQL dump file.

        Validates each SQL statement against an allowlist of safe
        command prefixes to prevent SQL injection attacks via
        tampered import files.

        Args:
            import_file: Path to SQL dump file

        Returns:
            True if successful
        """
        if not os.path.exists(import_file):
            logger.error(f"Import file not found: {import_file}")
            return False

        try:
            db_file = os.path.join(CHROMA_DB_DIR, "chroma.sqlite3")
            os.makedirs(CHROMA_DB_DIR, exist_ok=True)

            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()

            with open(import_file, 'r') as f:
                sql_script = f.read()

            # Parse and validate each statement individually
            sql_statements = re.split(r';\s*\n', sql_script)
            for stmt in sql_statements:
                stmt = stmt.strip()
                if not stmt:
                    continue
                if not StorageManager._validate_sql_statement(stmt):
                    error_msg = (
                        f"Blocked disallowed SQL statement during import: "
                        f"{stmt[:80]}... "
                        f"Allowed: {', '.join(_ALLOWED_SQL_PREFIXES)}"
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)

            cursor.executescript(sql_script)
            conn.commit()
            conn.close()

            logger.info(f"Database imported from: {import_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to import database: {e}")
            return False
    
    @staticmethod
    def cleanup_old_logs(max_age_days: int = 30) -> int:
        """Delete log files older than max_age_days.
        
        Args:
            max_age_days: Maximum age of logs to keep
            
        Returns:
            Number of files deleted
        """
        if not os.path.exists(LOG_DIR):
            return 0
        
        import time
        current_time = time.time()
        max_age_seconds = max_age_days * 86400
        deleted = 0
        
        try:
            for fname in os.listdir(LOG_DIR):
                if fname.endswith(".log"):
                    fpath = os.path.join(LOG_DIR, fname)
                    file_age = current_time - os.path.getmtime(fpath)
                    
                    if file_age > max_age_seconds:
                        try:
                            os.remove(fpath)
                            deleted += 1
                        except OSError as e:
                            logger.warning(f"Failed to delete old log: {e}")
            
            logger.info(f"Deleted {deleted} old log files")
        except OSError as e:
            logger.error(f"Failed to cleanup logs: {e}")
        
        return deleted


def _get_dir_size(path: str) -> int:
    """Get total size of directory and all contents in bytes."""
    total_size = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file():
                total_size += entry.stat().st_size
            elif entry.is_dir():
                total_size += _get_dir_size(entry.path)
    except OSError:
        pass
    return total_size


# Global storage manager instance
_storage_manager: Optional[StorageManager] = None


def get_storage_manager() -> StorageManager:
    """Get storage manager instance."""
    return StorageManager()
