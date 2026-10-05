# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Transcript Writer
Durably persists every VLM image analysis to disk with a timestamp.

Design goals:
  * Never lose a description - each record is flushed and fsync'd before the
    write call returns, so a hard kill loses at most the in-flight record.
  * Never break the video pipeline - all disk errors are caught, counted and
    logged; the caller always gets control back.
  * Never block the event loop - blocking I/O runs in a worker thread, and
    writes are serialized by an asyncio lock so record order is preserved.
  * Survive corruption - the default JSONL format is line-oriented, so a torn
    trailing line never invalidates the records before it. Periodic backup
    snapshots are taken every Nth record via a temp-file + atomic rename.
"""

import asyncio
import json
import logging
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Formats supported for the on-disk transcript
FORMAT_JSONL = "jsonl"
FORMAT_TXT = "txt"
FORMATS = (FORMAT_JSONL, FORMAT_TXT)

# Default number of records between backup snapshots
DEFAULT_BACKUP_EVERY = 25
# Default number of backup snapshots to retain (0 = keep every snapshot)
DEFAULT_BACKUP_KEEP = 10
# Only log every Nth consecutive write failure to avoid flooding the log
ERROR_LOG_INTERVAL = 20


def default_transcript_dir() -> Path:
    """
    Return the OS-appropriate directory for storing transcripts.

    Linux/Unix: $XDG_DATA_HOME/process-vlm-webui/transcripts (~/.local/share/...)
    macOS:      ~/Library/Application Support/process-vlm-webui/transcripts
    Windows:    %APPDATA%/process-vlm-webui/transcripts
    """
    if os.name == "posix":
        if "darwin" in os.sys.platform.lower():
            base = Path.home() / "Library" / "Application Support" / "process-vlm-webui"
        else:
            base = (
                Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
                / "process-vlm-webui"
            )
    else:
        base = Path(os.environ.get("APPDATA", Path.home())) / "process-vlm-webui"

    return base / "transcripts"


def default_transcript_name(
    fmt: str = FORMAT_JSONL, now: Optional[datetime] = None, index: int = 0
) -> str:
    """
    Build a per-session transcript filename, e.g. session-20251217-142530.jsonl

    `index` disambiguates sessions that start within the same second; index 0
    produces the plain name.
    """
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    suffix = "" if index == 0 else f"-{index}"
    return f"session-{stamp}{suffix}.{fmt}"


def claim_transcript_path(
    directory: os.PathLike | str, fmt: str = FORMAT_JSONL, now: Optional[datetime] = None
) -> Path:
    """
    Atomically claim an unused transcript filename inside `directory`.

    The generated name only has second resolution, so two servers started in the
    same second would otherwise pick the identical name and interleave their
    records into one file with duplicate sequence numbers. O_CREAT|O_EXCL makes
    the claim race-free: exactly one process can win each name, and the loser
    moves on to the next index.

    Returns the claimed (empty) path. Raises OSError if the directory is
    unwritable, which the caller reports through TranscriptWriter.prepare().
    """
    directory = Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)

    for index in range(1000):
        candidate = directory / default_transcript_name(fmt, now, index)
        try:
            fd = os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            continue
        os.close(fd)
        return candidate

    raise OSError(f"Could not claim a transcript filename in {directory}")


def _fsync_dir(path: Path) -> None:
    """
    fsync a directory so that a rename/create inside it is durable.

    Not supported on every platform (notably Windows); failures are non-fatal
    because the file contents themselves were already fsync'd.
    """
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


class TranscriptWriter:
    """
    Append-only, crash-resistant writer for VLM analysis results.

    Every accepted record is written as a single line (JSONL by default),
    flushed and fsync'd. After every ``backup_every`` records the live file is
    snapshotted into a ``backups/`` subdirectory using a temp file plus an
    atomic rename, so a snapshot is never left half-written.
    """

    def __init__(
        self,
        path: os.PathLike | str,
        fmt: str = FORMAT_JSONL,
        backup_every: int = DEFAULT_BACKUP_EVERY,
        backup_keep: int = DEFAULT_BACKUP_KEEP,
        max_bytes: int = 0,
        fsync: bool = True,
        enabled: bool = True,
    ):
        """
        Args:
            path: Full path of the transcript file to append to.
            fmt: "jsonl" (structured, default) or "txt" (human readable).
            backup_every: Take a backup snapshot every N records (0 disables).
            backup_keep: Number of snapshots to retain (0 keeps all).
            max_bytes: Rotate the live file once it exceeds this size (0 disables).
            fsync: fsync after every record. Disable only if throughput matters
                more than durability.
            enabled: Start in recording state. Can be toggled at runtime.
        """
        if fmt not in FORMATS:
            raise ValueError(f"Unsupported transcript format: {fmt!r} (expected one of {FORMATS})")

        self.path = Path(path).expanduser().resolve()
        self.format = fmt
        self.backup_every = max(0, int(backup_every))
        self.backup_keep = max(0, int(backup_keep))
        self.max_bytes = max(0, int(max_bytes))
        self.fsync = bool(fsync)
        self.enabled = bool(enabled)

        self.backup_dir = self.path.parent / "backups"

        # Serializes writes so records land in the order they were produced
        self._lock = asyncio.Lock()

        # Counters exposed through stats()
        self.records_written = 0  # records durably appended
        self.records_skipped = 0  # records dropped because recording is off
        self.backups_written = 0
        self.rotations = 0
        self.errors = 0
        self._consecutive_errors = 0
        self._since_backup = 0
        self.last_error: Optional[str] = None
        self.last_record_at: Optional[float] = None
        self.last_backup_at: Optional[float] = None
        self.last_backup_path: Optional[str] = None

        self._ready = False

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def prepare(self) -> bool:
        """
        Create the transcript directory and verify the file is writable.

        Returns True when the transcript is usable. On failure the writer
        disables itself rather than raising, so a bad path can never stop the
        server from serving video.
        """
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.backup_every:
                self.backup_dir.mkdir(parents=True, exist_ok=True)
            # Touch the file so an unwritable location fails now, not mid-session
            with open(self.path, "a", encoding="utf-8"):
                pass
            self._ready = True
            logger.info(f"📝 Transcript: {self.path}")
            if self.backup_every:
                keep = "all" if not self.backup_keep else f"last {self.backup_keep}"
                logger.info(
                    f"   Backup every {self.backup_every} records → {self.backup_dir} (keep {keep})"
                )
            else:
                logger.info("   Backups disabled")
            return True
        except OSError as e:
            self._ready = False
            self.enabled = False
            self.errors += 1
            self.last_error = str(e)
            logger.error(f"❌ Cannot write transcript to {self.path}: {e}")
            logger.error("   Recording is disabled; the rest of the server is unaffected.")
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def write(self, record: dict[str, Any]) -> bool:
        """
        Append one analysis record. Never raises.

        Args:
            record: Fields to persist. ``ts``/``ts_epoch``/``seq`` are added
                automatically when absent.

        Returns:
            True if the record was durably written to disk.
        """
        if not self.enabled or not self._ready:
            self.records_skipped += 1
            return False

        async with self._lock:
            entry = self._stamp(record)
            try:
                await asyncio.to_thread(self._append_and_maybe_backup, entry)
            except Exception as e:  # pragma: no cover - defensive
                self._note_error(e)
                return False
            return True

    async def write_event(self, kind: str, **fields: Any) -> bool:
        """Append a non-analysis record (session start/stop, config change...)."""
        return await self.write({"kind": kind, **fields})

    async def backup_now(self) -> Optional[str]:
        """
        Force a backup snapshot immediately. Never raises.

        Returns the snapshot path, or None if no backup was taken.
        """
        if not self._ready:
            return None
        async with self._lock:
            try:
                return await asyncio.to_thread(self._snapshot)
            except Exception as e:  # pragma: no cover - defensive
                self._note_error(e)
                return None

    async def close(self) -> None:
        """Flush a final snapshot so a clean shutdown always leaves a backup."""
        if not self._ready:
            return
        await self.write_event(
            "session_end",
            records_written=self.records_written,
            backups_written=self.backups_written,
        )
        # Skip when writing session_end just tripped the interval - that
        # snapshot already contains everything, so a second is pure duplication
        if self.backup_every and self._since_backup:
            path = await self.backup_now()
            if path:
                logger.info(f"📝 Final transcript backup: {path}")
        logger.info(
            f"📝 Transcript closed: {self.records_written} records → {self.path}"
            f" ({self.errors} write errors)"
        )

    def set_enabled(self, enabled: bool) -> bool:
        """Pause or resume recording. Returns the resulting state."""
        want = bool(enabled)
        if want and not self._ready:
            # Retry setup - the target may have become writable since startup
            self.prepare()
        self.enabled = want and self._ready
        logger.info(f"📝 Transcript recording {'enabled' if self.enabled else 'paused'}")
        return self.enabled

    def stats(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of writer state (for API/UI)."""
        try:
            size = self.path.stat().st_size if self.path.exists() else 0
        except OSError:
            size = 0

        return {
            "enabled": self.enabled,
            "ready": self._ready,
            "path": str(self.path),
            "filename": self.path.name,
            "format": self.format,
            "records_written": self.records_written,
            "records_skipped": self.records_skipped,
            "bytes": size,
            "backup_every": self.backup_every,
            "backup_keep": self.backup_keep,
            "backups_written": self.backups_written,
            "backup_dir": str(self.backup_dir),
            "records_until_backup": (
                max(0, self.backup_every - self._since_backup) if self.backup_every else None
            ),
            "last_backup_at": self.last_backup_at,
            "last_backup_path": self.last_backup_path,
            "last_record_at": self.last_record_at,
            "rotations": self.rotations,
            "errors": self.errors,
            "last_error": self.last_error,
        }

    def brief_stats(self) -> dict[str, Any]:
        """Compact stats for high-frequency broadcast alongside VLM metrics."""
        return {
            "enabled": self.enabled,
            "records": self.records_written,
            "backups": self.backups_written,
            "errors": self.errors,
        }

    # ------------------------------------------------------------------
    # Internals (blocking - always called from a worker thread)
    # ------------------------------------------------------------------

    def _stamp(self, record: dict[str, Any]) -> dict[str, Any]:
        """Attach sequence number and timestamps to a record."""
        now = time.time()
        entry = {
            "seq": self.records_written + 1,
            "ts": datetime.fromtimestamp(now, tz=timezone.utc)
            .astimezone()
            .isoformat(timespec="milliseconds"),
            "ts_epoch": round(now, 3),
            "kind": "analysis",
        }
        entry.update(record)
        return entry

    def _format_record(self, entry: dict[str, Any]) -> str:
        """Render one record as a single line of text (always newline-terminated)."""
        if self.format == FORMAT_JSONL:
            # default=str keeps unexpected types from raising mid-write
            return json.dumps(entry, ensure_ascii=False, default=str) + "\n"

        # Human-readable single-line text format
        parts = [f"[{entry.get('ts', '')}]", f"#{entry.get('seq', '')}"]
        if entry.get("kind") and entry["kind"] != "analysis":
            parts.append(f"({entry['kind']})")
        if entry.get("source"):
            parts.append(f"<{entry['source']}>")
        if entry.get("model"):
            parts.append(f"{entry['model']}")
        if entry.get("latency_ms") is not None:
            parts.append(f"{float(entry['latency_ms']):.0f}ms")
        text = entry.get("response") or entry.get("message") or ""
        # Collapse newlines so one record stays one line
        text = " ".join(str(text).split())
        return " ".join(parts) + (f" | {text}" if text else "") + "\n"

    def _append_and_maybe_backup(self, entry: dict[str, Any]) -> None:
        """Durably append one record, then rotate/snapshot if it is time."""
        line = self._format_record(entry)

        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            if self.fsync:
                os.fsync(f.fileno())

        self.records_written += 1
        self.last_record_at = entry.get("ts_epoch", time.time())
        self._consecutive_errors = 0
        self._since_backup += 1

        if self.backup_every and self._since_backup >= self.backup_every:
            self._snapshot()

        if self.max_bytes:
            self._rotate_if_needed()

    def _unique_path(self, directory: Path, infix: str) -> Path:
        """
        Build an archive path that cannot clobber an existing file.

        Timestamps only have second resolution, so two archives created within
        the same second would otherwise collide - and an os.replace() onto an
        existing archive silently destroys it.
        """
        candidate = directory / f"{self.path.stem}.{infix}{self.path.suffix}"
        counter = 1
        while candidate.exists():
            candidate = directory / f"{self.path.stem}.{infix}-{counter}{self.path.suffix}"
            counter += 1
        return candidate

    def _snapshot(self) -> Optional[str]:
        """
        Copy the live transcript into backups/ atomically.

        Writes to a temp file in the destination directory, fsyncs it, then
        renames it into place, so an interrupted backup can never overwrite a
        good snapshot with a partial one.
        """
        if not self.path.exists() or self.path.stat().st_size == 0:
            self._since_backup = 0
            return None

        self.backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        dest = self._unique_path(self.backup_dir, f"{stamp}.n{self.records_written}")

        tmp_fd, tmp_name = tempfile.mkstemp(dir=str(self.backup_dir), prefix=".tmp-backup-")
        os.close(tmp_fd)
        try:
            shutil.copyfile(self.path, tmp_name)
            with open(tmp_name, "rb") as f:
                os.fsync(f.fileno())
            os.replace(tmp_name, dest)
            _fsync_dir(self.backup_dir)
        except OSError:
            # Clean up the temp file, then let the caller record the failure
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

        self._since_backup = 0
        self.backups_written += 1
        self.last_backup_at = time.time()
        self.last_backup_path = str(dest)
        logger.info(f"📝 Transcript backup #{self.backups_written}: {dest.name}")

        self._prune_backups()
        return str(dest)

    def _prune_backups(self) -> None:
        """Delete the oldest snapshots beyond the retention limit."""
        if not self.backup_keep:
            return
        try:
            pattern = f"{self.path.stem}.*{self.path.suffix}"
            snapshots = sorted(
                (p for p in self.backup_dir.glob(pattern) if p.is_file()),
                key=lambda p: p.stat().st_mtime,
            )
        except OSError:
            return

        for old in snapshots[: max(0, len(snapshots) - self.backup_keep)]:
            try:
                old.unlink()
                logger.debug(f"Pruned old transcript backup: {old.name}")
            except OSError as e:
                logger.debug(f"Could not prune backup {old}: {e}")

    def _rotate_if_needed(self) -> None:
        """Move the live file aside once it grows past max_bytes."""
        try:
            size = self.path.stat().st_size
        except OSError:
            return
        if size < self.max_bytes:
            return

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        rotated = self._unique_path(self.path.parent, f"{stamp}.n{self.records_written}")
        try:
            os.replace(self.path, rotated)
            _fsync_dir(self.path.parent)
        except OSError as e:
            logger.warning(f"Transcript rotation failed: {e}")
            return

        self.rotations += 1
        self._since_backup = 0
        logger.info(f"📝 Transcript rotated at {size} bytes → {rotated.name}")

    def _note_error(self, exc: Exception) -> None:
        """Record a write failure without ever propagating it to the caller."""
        self.errors += 1
        self._consecutive_errors += 1
        self.last_error = f"{type(exc).__name__}: {exc}"

        if self._consecutive_errors == 1 or self._consecutive_errors % ERROR_LOG_INTERVAL == 0:
            logger.error(f"❌ Transcript write failed ({self._consecutive_errors} in a row): {exc}")
            if self._consecutive_errors == 1:
                logger.error(f"   Target: {self.path} - analysis results may be lost")
