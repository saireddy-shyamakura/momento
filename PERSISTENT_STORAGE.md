# Momento Persistent Storage & Recovery Guide

## Overview

Momento uses a sophisticated storage system to ensure your data persists reliably across sessions. This guide covers how data is stored, managed, backed up, and recovered.

---

## Storage Architecture

### Directory Structure

```
~/.local/share/momento/                    # Main data directory (XDG standard)
├── chroma_db/                             # Vector database (ChromaDB)
│   ├── chroma.sqlite3                     # Main database file (~100 MB per 100k items)
│   └── [collection-uuid]/                 # Collection metadata directories
│       ├── data/
│       ├── index/
│       └── metadata.json
├── embedding_cache/                       # Extracted embeddings cache
│   ├── image_cache.pkl                    # Image embeddings
│   ├── text_cache.pkl                     # Text embeddings
│   └── cache_index.json                   # Cache metadata
├── logs/                                  # Application logs
│   ├── momento_YYYY-MM-DD.log             # Daily log files
│   └── error.log                          # Error log
└── indexing_checkpoint.json               # Recovery checkpoint
```

### Configuration Directory

```
~/.config/momento/                         # User configuration
├── config.toml                            # Main configuration file
└── config.backup.toml                     # Backup of previous config
```

---

## Data Persistence Layers

### Layer 1: Vector Database (ChromaDB)

**Persistent Storage of:**
- Vector embeddings (512-768 dimensional vectors)
- Metadata (file paths, timestamps, source types)
- Collection information
- Indexing state

**Storage Format:**
```json
{
  "id": "path/to/image.jpg|||aug_0",
  "embedding": [0.234, -0.156, 0.889, ...],
  "metadata": {
    "file_path": "path/to/image.jpg",
    "type": "image",
    "augmentation": 0,
    "timestamp": "2025-05-30T10:15:00Z",
    "model": "ViT-B/16"
  },
  "document": "OCR extracted text or description"
}
```

**Durability:**
- SQLite with WAL (Write-Ahead Logging) for crash safety
- ACID transactions ensure data consistency
- Automatic syncing to disk after each batch

**Storage Size Estimation:**
```
Base embedding (512-dim):      ~2 KB
Multi-embed (5x):             ~10 KB
Video frame:                  ~2 KB
Detected object:              ~2 KB  
OCR text:                     ~1-10 KB

Total per image:              ~15-40 KB (with all features)
Total per 1000 images:        ~15-40 MB
Total per 100k images:        ~1.5-4 GB
```

### Layer 2: Embedding Cache (LRU)

**Purpose:** Speed up re-indexing by caching computed embeddings

**Storage:**
```
~/.local/share/momento/embedding_cache/
├── image_cache.pkl           # Pickled image embeddings (compressed)
├── text_cache.pkl            # Pickled text embeddings (compressed)
└── cache_index.json          # File-hash to embedding mapping
```

**Cache Index Structure:**
```json
{
  "md5:abc123...": {
    "model": "ViT-B/16",
    "timestamp": "2025-05-30T10:15:00Z",
    "size_kb": 2
  }
}
```

**Features:**
- **LRU Eviction:** Oldest unused entries removed when cache grows
- **Hash-Based Lookup:** Fast retrieval using file content hash
- **Model-Aware:** Different embeddings for different models
- **Compression:** Numpy arrays stored in compressed pickle format

**Configuration:**
```toml
# ~/.config/momento/config.toml
[cache]
cache_max_size_gb = 5         # Default: 5 GB limit
cache_eviction_policy = "lru" # Least Recently Used
```

**Management Commands:**
```bash
# View cache size
du -sh ~/.local/share/momento/embedding_cache/

# Clear all cache
momento cache clear

# View cache metadata
cat ~/.local/share/momento/embedding_cache/cache_index.json | jq

# Partial clear (example - when implemented)
momento cache clear --older-than 30d
momento cache clear --model "ViT-B/32"
```

### Layer 3: Checkpoint & Recovery

**Purpose:** Resume indexing after interruption (crash, power loss, Ctrl+C)

**Checkpoint File:**
```
~/.local/share/momento/indexing_checkpoint.json
```

**Checkpoint Content:**
```json
{
  "folder": "/home/user/pictures",
  "collection_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "timestamp": "2025-05-30T10:15:00Z",
  "status": "in_progress",
  "features_status": {
    "images": {
      "status": "completed",
      "count": 250,
      "processed_files": ["photo1.jpg", "photo2.jpg", ...]
    },
    "multi_embed": {
      "status": "completed",
      "augmentation_count": 5
    },
    "videos": {
      "status": "in_progress",
      "count": 2,
      "processed_files": ["video1.mp4"],
      "current_file": "video2.mp4",
      "current_frame": 23
    },
    "yolo": {
      "status": "pending",
      "count": 0
    },
    "ocr": {
      "status": "pending",
      "count": 0
    }
  },
  "config_snapshot": {
    "model_name": "ViT-B/16",
    "enable_multi_embed": true,
    "enable_video_indexing": true,
    "enable_yolo": false,
    "enable_ocr": true
  }
}
```

**Recovery Behavior:**

1. **Automatic Detection:** On next `momento index`, checkpoint is detected
2. **Validation:** Config snapshot verified against current settings
3. **Resumption:**
   - Successfully completed features skipped
   - In-progress feature resumes from last position
   - Pending features start fresh

4. **Cleanup:** Checkpoint deleted on successful completion

**Example Scenario:**
```
Indexing interrupted at frame 23 of video2.mp4:
$ momento index ~/pictures  # First attempt - interrupted
Indexing interrupted. Run again to resume.

$ momento index ~/pictures  # Resume
[Loading checkpoint] ~/pictures
[Images] Already processed - skipping
[Multi-embed] Already processed - skipping  
[Videos] Resuming from video2.mp4:frame 23 ✓
[YOLO] Processing detected objects...
[OCR] Extracting text...
```

**Force Fresh Start (Clear Checkpoint):**
```bash
rm ~/.local/share/momento/indexing_checkpoint.json
momento index ~/pictures  # Starts fresh, no resume
```

### Layer 4: Logging

**Location:** `~/.local/share/momento/logs/`

**Log Files:**
```
momento_2025-05-30.log      # Daily log with all events
error.log                   # Errors only
```

**Logging Configuration:**
```toml
# ~/.config/momento/config.toml
[logging]
log_format = "text"       # "text" or "json"
log_level = "INFO"        # DEBUG, INFO, WARNING, ERROR
```

**Log Structure (JSON format):**
```json
{
  "timestamp": "2025-05-30T10:15:23.456Z",
  "level": "INFO",
  "module": "indexer",
  "event": "image_added",
  "file_path": "photo1.jpg",
  "embedding_model": "ViT-B/16",
  "duration_ms": 145,
  "memory_mb": 2048
}
```

---

## Backup & Recovery Procedures

### Manual Backup

**Full Backup (Entire data directory):**
```bash
# Create timestamped backup
mkdir -p ~/backups
tar -czf ~/backups/momento_backup_$(date +%Y%m%d_%H%M%S).tar.gz \
  ~/.local/share/momento/

# Or simple copy
cp -r ~/.local/share/momento ~/momento_backup_$(date +%Y%m%d)

# Verify backup
ls -lh ~/backups/
du -sh ~/momento_backup_*
```

**Selective Backup (Database only):**
```bash
# Backup only database (smallest, fastest)
cp ~/.local/share/momento/chroma_db/chroma.sqlite3 \
  ~/backups/chroma_$(date +%Y%m%d).db

# Backup database + config
tar -czf ~/backups/momento_config_db_$(date +%Y%m%d).tar.gz \
  ~/.local/share/momento/chroma_db/ \
  ~/.config/momento/

# Exclude cache (can be regenerated)
tar -czf ~/backups/momento_no_cache_$(date +%Y%m%d).tar.gz \
  --exclude='embedding_cache' \
  ~/.local/share/momento/
```

**Automated Backup Script:**
```bash
#!/bin/bash
# backup_momento.sh

BACKUP_DIR="$HOME/backups/momento"
RETENTION_DAYS=30

mkdir -p "$BACKUP_DIR"

# Create backup
BACKUP_FILE="$BACKUP_DIR/backup_$(date +%Y%m%d_%H%M%S).tar.gz"
tar -czf "$BACKUP_FILE" \
  ~/.local/share/momento/chroma_db/ \
  ~/.config/momento/

echo "Created backup: $BACKUP_FILE"

# Remove old backups (>30 days)
find "$BACKUP_DIR" -name "backup_*.tar.gz" -mtime +$RETENTION_DAYS -delete
echo "Cleaned old backups (older than $RETENTION_DAYS days)"
```

**Schedule with cron:**
```bash
# Backup every day at 2 AM
0 2 * * * /home/user/backup_momento.sh
```

### Restore from Backup

**Full Restore (Complete recovery):**
```bash
# Stop any running momento processes
pkill -f momento

# Remove current data
rm -rf ~/.local/share/momento ~/.config/momento

# Restore from backup
tar -xzf ~/backups/momento_backup_20250530.tar.gz -C ~/

# Verify restore
ls -la ~/.local/share/momento/chroma_db/

# Restart
momento index ~/pictures
```

**Selective Restore (Database only):**
```bash
# Restore just database
mkdir -p ~/.local/share/momento/chroma_db
cp ~/backups/chroma_20250530.db \
  ~/.local/share/momento/chroma_db/chroma.sqlite3

# Verify
sqlite3 ~/.local/share/momento/chroma_db/chroma.sqlite3 ".tables"
```

**Restore to Different Location:**
```bash
# Useful for testing/verification
mkdir -p ~/test_restore
tar -xzf ~/backups/momento_backup_20250530.tar.gz -C ~/test_restore

# Test the restored data
cd ~/test_restore
momento config show  # Check config
```

### Point-in-Time Recovery

**Database Export (SQL dump):**
```bash
# Export database schema and data
sqlite3 ~/.local/share/momento/chroma_db/chroma.sqlite3 .dump > \
  ~/backups/momento_schema_$(date +%Y%m%d).sql

# Restore from SQL dump
sqlite3 ~/.local/share/momento/chroma_db/chroma.sqlite3 < \
  ~/backups/momento_schema_20250530.sql
```

---

## Advanced Storage Management

### Monitor Storage Usage

```bash
# Overall storage
du -sh ~/.local/share/momento/

# Breakdown by component
du -sh ~/.local/share/momento/chroma_db/
du -sh ~/.local/share/momento/embedding_cache/
du -sh ~/.local/share/momento/logs/

# Detailed breakdown
du -sh ~/.local/share/momento/* | sort -h

# Database size specifically
ls -lh ~/.local/share/momento/chroma_db/chroma.sqlite3

# Storage per collection
du -sh ~/.local/share/momento/chroma_db/*/
```

### Optimize Storage

**Clear Cache (safe to delete, regenerated on next indexing):**
```bash
# Remove cache (saves ~1-5 GB typically)
rm -rf ~/.local/share/momento/embedding_cache/*
mkdir -p ~/.local/share/momento/embedding_cache

# Or use CLI
momento cache clear
```

**Archive Old Logs:**
```bash
# Compress old log files
gzip ~/.local/share/momento/logs/momento_2025-05*.log

# Remove logs older than 30 days
find ~/.local/share/momento/logs -name "*.log" -mtime +30 -delete
```

**Compact Database:**
```bash
# SQLite VACUUM optimizes storage
sqlite3 ~/.local/share/momento/chroma_db/chroma.sqlite3 "VACUUM;"

# Check before/after
ls -lh ~/.local/share/momento/chroma_db/chroma.sqlite3
```

**Remove Specific Collections:**
```bash
# List collections
sqlite3 ~/.local/share/momento/chroma_db/chroma.sqlite3 \
  "SELECT name FROM sqlite_master WHERE type='table';"

# Delete collection (caution!)
sqlite3 ~/.local/share/momento/chroma_db/chroma.sqlite3 \
  "DELETE FROM embeddings WHERE collection_id='specific-uuid';"

# Compact after deletion
sqlite3 ~/.local/share/momento/chroma_db/chroma.sqlite3 "VACUUM;"
```

### Database Inspection

**View Database Statistics:**
```bash
sqlite3 ~/.local/share/momento/chroma_db/chroma.sqlite3 << EOF
-- Count embeddings
SELECT COUNT(*) as total_embeddings FROM embeddings;

-- Count by type
SELECT metadata ->> 'type' as type, COUNT(*) as count 
FROM embeddings 
GROUP BY metadata ->> 'type';

-- Largest tables
SELECT name, page_count * page_size / 1024 / 1024 as size_mb 
FROM pragma_page_count(), pragma_page_size(), sqlite_master 
WHERE type='table' 
ORDER BY size_mb DESC;

-- Database size
SELECT page_count * page_size / 1024 / 1024 as size_mb 
FROM pragma_page_count(), pragma_page_size();
EOF
```

**Export Collection Metadata:**
```bash
# Export all metadata as JSON
sqlite3 ~/.local/share/momento/chroma_db/chroma.sqlite3 \
  -json "SELECT id, metadata, document FROM embeddings LIMIT 100;" > \
  ~/collections_metadata.json
```

---

## Disaster Recovery

### Total Data Loss Recovery

**Scenario:** Database corrupted, cannot recover

**Steps:**
```bash
# 1. Verify backup exists
ls -lh ~/backups/momento_backup_*.tar.gz

# 2. Remove corrupted data
rm -rf ~/.local/share/momento

# 3. Restore from latest backup
tar -xzf ~/backups/momento_backup_latest.tar.gz -C ~/

# 4. Verify restoration
moment config show
momento search "test query"

# 5. If backup is outdated, re-index from media
momento index ~/pictures
```

### Partial Corruption Recovery

**Scenario:** Some embeddings corrupted, but media files intact

**Steps:**
```bash
# 1. Identify corrupted data
sqlite3 ~/.local/share/momento/chroma_db/chroma.sqlite3 \
  "SELECT COUNT(*) FROM embeddings WHERE embedding IS NULL;"

# 2. Remove corrupted records
sqlite3 ~/.local/share/momento/chroma_db/chroma.sqlite3 \
  "DELETE FROM embeddings WHERE embedding IS NULL;"

# 3. Compact database
sqlite3 ~/.local/share/momento/chroma_db/chroma.sqlite3 "VACUUM;"

# 4. Clear checkpoint to force fresh index
rm ~/.local/share/momento/indexing_checkpoint.json

# 5. Re-index to rebuild missing embeddings
momento index ~/pictures
```

### Migration Between Systems

**Move to New Computer:**
```bash
# On old computer
tar -czf ~/momento_migration_$(date +%Y%m%d).tar.gz \
  ~/.local/share/momento/ \
  ~/.config/momento/

# Transfer file to new computer (scp, rsync, USB, etc.)
scp ~/momento_migration_*.tar.gz user@newhost:~/

# On new computer
tar -xzf ~/momento_migration_*.tar.gz -C ~/

# Verify
momento config show
momento search "test query"
```

---

## Performance Optimization

### Optimize for Storage Space
```toml
# ~/.config/momento/config.toml
enable_multi_embed = false        # Reduces embeddings by 5x
augmentation_count = 1            # Instead of default 5
cache_max_size_gb = 1             # Smaller cache
log_level = "WARNING"             # Fewer logs
```

### Optimize for Speed
```toml
# ~/.config/momento/config.toml
model_name = "ViT-B/32"           # Smallest, fastest model
enable_multi_embed = false        # Faster indexing
enable_video_indexing = false     # Skip video processing
enable_yolo = false               # Skip object detection
enable_ocr = false                # Skip text extraction
cache_max_size_gb = 10            # Larger cache for faster re-index
```

### Optimize for Search Quality
```toml
# ~/.config/momento/config.toml
model_name = "ViT-L/14"           # Highest quality embeddings
enable_multi_embed = true         # Better recall
augmentation_count = 5            # Maximum augmentation
enable_yolo = true                # Fine-grained search
enable_ocr = true                 # Text-based search
```

---

## Troubleshooting Storage Issues

### "Not enough disk space"
```bash
# Check available space
df -h ~/.local/share/

# Clear cache (fastest recovery)
momento cache clear

# Check what's using space
du -sh ~/.local/share/momento/*

# Remove old logs
rm ~/.local/share/momento/logs/*.log
```

### "Index corrupted"
```bash
# Restore from checkpoint if available
momento index ~/pictures  # May auto-recover

# Or restore from backup
tar -xzf ~/backups/momento_backup_latest.tar.gz -C ~/
```

### "Embeddings cache keeps growing"
```bash
# Clear cache periodically
momento cache clear

# Or reduce cache limit
# Edit ~/.config/momento/config.toml
cache_max_size_gb = 2  # Lower limit

# Check cache size
du -sh ~/.local/share/momento/embedding_cache/
```

### "Database queries are slow"
```bash
# Optimize database
sqlite3 ~/.local/share/momento/chroma_db/chroma.sqlite3 << EOF
VACUUM;
ANALYZE;
PRAGMA optimize;
EOF
```

---

## Checklist: Backup Strategy

- [ ] Identify backup location (external drive, cloud storage, etc.)
- [ ] Set backup frequency (daily, weekly, monthly)
- [ ] Test backup script/process
- [ ] Document restore procedure
- [ ] Schedule automated backups via cron
- [ ] Monitor backup completion
- [ ] Store backups in multiple locations
- [ ] Verify backup integrity periodically
- [ ] Document storage requirements

---

## See Also

- [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) - Full customization guide
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Quick reference for common tasks
- [README.md](README.md) - Project overview

