"""SHA256-based LLM response cache.

Caches API responses by (prompt_hash, model) to avoid redundant calls.
Stored as JSON files in a .autolean_cache/ directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

from .util import CommandResult

_DEFAULT_CACHE_DIR = ".autolean_cache"


class ResponseCache:
    """Disk-backed cache for LLM API responses."""

    def __init__(self, cache_dir: Optional[Path] = None, *, enabled: bool = True):
        self.enabled = enabled
        self.cache_dir = cache_dir or Path(_DEFAULT_CACHE_DIR)
        self._hits = 0
        self._misses = 0

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    @staticmethod
    def _cache_key(prompt: str, model: str) -> str:
        """Generate a deterministic cache key from prompt + model."""
        content = json.dumps({"prompt": prompt, "model": model}, sort_keys=True)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        # Use first 2 chars as subdirectory to avoid flat directory with thousands of files
        subdir = self.cache_dir / key[:2]
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir / f"{key}.json"

    def get(self, prompt: str, model: str) -> Optional[CommandResult]:
        """Look up a cached response. Returns None on miss."""
        if not self.enabled:
            return None

        key = self._cache_key(prompt, model)
        path = self._cache_path(key)

        if not path.exists():
            self._misses += 1
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._hits += 1
            return CommandResult(
                argv=data.get("argv", []),
                returncode=data.get("returncode", 0),
                stdout=data.get("stdout", ""),
                stderr=data.get("stderr", ""),
            )
        except (json.JSONDecodeError, OSError, KeyError):
            self._misses += 1
            return None

    def put(self, prompt: str, model: str, result: CommandResult) -> None:
        """Store a successful response in the cache."""
        if not self.enabled:
            return
        # Only cache successful responses
        if result.returncode != 0:
            return

        key = self._cache_key(prompt, model)
        path = self._cache_path(key)

        data = {
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "model": model,
            "cached_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "argv": result.argv,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        try:
            # Atomic write: write to temp file then rename to prevent corruption
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=True)
                os.replace(tmp_path, path)
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except OSError:
            pass  # Cache write failure is non-fatal

    def clear(self) -> int:
        """Remove all cached entries. Returns count of files removed."""
        if not self.cache_dir.exists():
            return 0
        count = 0
        for f in self.cache_dir.rglob("*.json"):
            try:
                f.unlink()
                count += 1
            except OSError:
                pass
        return count

    def stats(self) -> dict[str, int]:
        """Return cache hit/miss statistics."""
        total = 0
        if self.cache_dir.exists():
            total = sum(1 for _ in self.cache_dir.rglob("*.json"))
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total_entries": total,
        }
