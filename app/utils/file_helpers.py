"""utils/file_helpers.py — shared file utilities."""

import hashlib
from pathlib import Path


def compute_file_hash(content: bytes) -> str:
    """SHA-256 hash of file content — useful for dedup detection."""
    return hashlib.sha256(content).hexdigest()


def human_readable_size(size_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes //= 1024
    return f"{size_bytes:.1f} TB"


def safe_filename(filename: str) -> str:
    """Strip path traversal characters from a filename."""
    return Path(filename).name.replace("..", "").replace("/", "").replace("\\", "")
