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

"""Unit tests for the transcript writer (persisting VLM analyses to disk)."""

import asyncio
import json
from datetime import datetime

import pytest

from process_vlm_webui.transcript import (
    FORMAT_JSONL,
    FORMAT_TXT,
    TranscriptWriter,
    claim_transcript_path,
    default_transcript_dir,
    default_transcript_name,
)


def read_records(path):
    """Parse a JSONL transcript into a list of dicts."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture
def writer(tmp_path):
    """A prepared writer with backups every 5 records."""
    w = TranscriptWriter(tmp_path / "session.jsonl", backup_every=5, backup_keep=3)
    assert w.prepare() is True
    return w


class TestWriting:
    """Records land on disk with timestamps and sequence numbers."""

    async def test_writes_record_with_timestamp_and_sequence(self, writer):
        await writer.write({"response": "a cat on a couch", "model": "llava:7b"})

        records = read_records(writer.path)
        assert len(records) == 1
        rec = records[0]
        assert rec["response"] == "a cat on a couch"
        assert rec["model"] == "llava:7b"
        assert rec["seq"] == 1
        assert rec["kind"] == "analysis"
        assert isinstance(rec["ts_epoch"], float)
        # ts must be a parseable ISO-8601 timestamp carrying a UTC offset
        parsed = datetime.fromisoformat(rec["ts"])
        assert parsed.tzinfo is not None

    async def test_every_analysis_is_recorded_in_order(self, writer):
        for i in range(20):
            await writer.write({"response": f"description {i}"})

        records = read_records(writer.path)
        assert len(records) == 20
        assert [r["seq"] for r in records] == list(range(1, 21))
        assert [r["response"] for r in records] == [f"description {i}" for i in range(20)]
        assert writer.records_written == 20

    async def test_concurrent_writes_are_serialized(self, writer):
        await asyncio.gather(*(writer.write({"response": f"r{i}"}) for i in range(50)))

        records = read_records(writer.path)
        assert len(records) == 50
        # Sequence numbers are unique and gapless even under concurrency
        assert sorted(r["seq"] for r in records) == list(range(1, 51))

    async def test_appends_to_existing_file(self, tmp_path):
        path = tmp_path / "session.jsonl"
        first = TranscriptWriter(path, backup_every=0)
        first.prepare()
        await first.write({"response": "first run"})

        second = TranscriptWriter(path, backup_every=0)
        second.prepare()
        await second.write({"response": "second run"})

        records = read_records(path)
        assert [r["response"] for r in records] == ["first run", "second run"]

    async def test_unicode_and_newlines_survive_roundtrip(self, writer):
        text = "a café ☕ sign\nwith a second line"
        await writer.write({"response": text})

        records = read_records(writer.path)
        assert records[0]["response"] == text
        # One record is always exactly one line, so a torn line stays isolated
        assert writer.path.read_text(encoding="utf-8").count("\n") == 1

    async def test_non_serializable_values_do_not_break_the_write(self, writer):
        await writer.write({"response": "ok", "extra": object()})

        records = read_records(writer.path)
        assert len(records) == 1
        assert records[0]["response"] == "ok"

    async def test_write_event_records_session_markers(self, writer):
        await writer.write_event("session_start", model="llava:7b")

        rec = read_records(writer.path)[0]
        assert rec["kind"] == "session_start"
        assert rec["model"] == "llava:7b"


class TestBackups:
    """A snapshot is taken every Nth write and old ones are pruned."""

    async def test_backup_every_nth_write(self, writer):
        for i in range(4):
            await writer.write({"response": f"r{i}"})
        assert writer.backups_written == 0, "no backup before the threshold"

        await writer.write({"response": "r4"})  # 5th record
        assert writer.backups_written == 1

        snapshots = list(writer.backup_dir.glob("session.*.jsonl"))
        assert len(snapshots) == 1
        # The snapshot holds every record written so far
        assert len(read_records(snapshots[0])) == 5

    async def test_backups_repeat_on_each_interval(self, writer):
        for i in range(15):
            await writer.write({"response": f"r{i}"})

        assert writer.backups_written == 3
        assert len(read_records(writer.path)) == 15

    async def test_backup_retention_prunes_oldest(self, writer):
        # backup_keep=3, backup_every=5 -> 5 snapshots taken, 3 retained
        for i in range(25):
            await writer.write({"response": f"r{i}"})

        assert writer.backups_written == 5
        snapshots = sorted(writer.backup_dir.glob("session.*.jsonl"))
        assert len(snapshots) == 3
        # The retained snapshots are the newest ones (20 and 25 records)
        assert len(read_records(snapshots[-1])) == 25

    async def test_keep_all_backups_when_retention_disabled(self, tmp_path):
        w = TranscriptWriter(tmp_path / "s.jsonl", backup_every=2, backup_keep=0)
        w.prepare()
        for i in range(10):
            await w.write({"response": f"r{i}"})

        assert w.backups_written == 5
        assert len(list(w.backup_dir.glob("s.*.jsonl"))) == 5

    async def test_backup_disabled(self, tmp_path):
        w = TranscriptWriter(tmp_path / "s.jsonl", backup_every=0)
        w.prepare()
        for i in range(30):
            await w.write({"response": f"r{i}"})

        assert w.backups_written == 0
        assert not w.backup_dir.exists()
        assert len(read_records(w.path)) == 30

    async def test_backup_now_forces_a_snapshot(self, writer):
        await writer.write({"response": "only one"})
        path = await writer.backup_now()

        assert path is not None
        assert len(read_records(path)) == 1
        # The counter resets, so the next automatic backup is a full interval away
        assert writer.stats()["records_until_backup"] == 5

    async def test_backup_now_on_empty_transcript_is_a_noop(self, writer):
        assert await writer.backup_now() is None
        assert writer.backups_written == 0

    async def test_no_partial_snapshots_left_behind(self, writer):
        for i in range(12):
            await writer.write({"response": f"r{i}"})

        leftovers = list(writer.backup_dir.glob(".tmp-backup-*"))
        assert leftovers == [], "temp files must be renamed or removed"

    async def test_close_writes_final_snapshot(self, writer):
        await writer.write({"response": "important last description"})
        await writer.close()

        snapshots = list(writer.backup_dir.glob("session.*.jsonl"))
        assert len(snapshots) == 1
        records = read_records(snapshots[0])
        # Both the analysis and the session_end marker are captured
        assert len(records) == 2
        assert records[0]["response"] == "important last description"
        # The marker reports the record count as of the moment it was emitted
        assert records[-1]["kind"] == "session_end"
        assert records[-1]["records_written"] == 1

    async def test_close_does_not_duplicate_a_just_taken_snapshot(self, tmp_path):
        # backup_every=2: record 1 + the session_end marker trip the interval,
        # so close() must not add a second identical snapshot
        w = TranscriptWriter(tmp_path / "session.jsonl", backup_every=2, backup_keep=5)
        w.prepare()
        await w.write({"response": "only analysis"})
        await w.close()

        assert w.backups_written == 1
        snapshots = list(w.backup_dir.glob("session.*.jsonl"))
        assert len(snapshots) == 1
        assert len(read_records(snapshots[0])) == 2

    async def test_close_snapshots_records_written_since_the_last_backup(self, writer):
        # backup_every=5: 6 records leaves 1 unsaved, plus the session_end marker
        for i in range(6):
            await writer.write({"response": f"r{i}"})
        assert writer.backups_written == 1

        await writer.close()
        assert writer.backups_written == 2
        newest = max(writer.backup_dir.glob("session.*.jsonl"), key=lambda p: p.stat().st_mtime)
        assert len(read_records(newest)) == 7


class TestRotation:
    """The live file is capped when max_bytes is set."""

    async def test_rotates_past_max_bytes(self, tmp_path):
        w = TranscriptWriter(tmp_path / "s.jsonl", backup_every=0, max_bytes=500)
        w.prepare()
        for i in range(20):
            await w.write({"response": "x" * 80})

        assert w.rotations > 0
        rotated = list(tmp_path.glob("s.*.jsonl"))
        assert rotated, "rotated files are kept alongside the live file"
        # No records were lost across rotations
        total = len(read_records(w.path)) + sum(len(read_records(p)) for p in rotated)
        assert total == 20


class TestRobustness:
    """Disk problems are contained: recording degrades, inference does not."""

    async def test_unwritable_path_disables_recording_without_raising(self, tmp_path):
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")

        w = TranscriptWriter(blocker / "nested" / "s.jsonl")
        assert w.prepare() is False
        assert w.enabled is False
        # Writing is a no-op rather than an exception
        assert await w.write({"response": "dropped"}) is False
        assert w.records_skipped == 1

    async def test_write_failure_is_counted_not_raised(self, writer, monkeypatch):
        def boom(*args, **kwargs):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr("builtins.open", boom)
        assert await writer.write({"response": "lost"}) is False
        assert writer.errors == 1
        assert "No space left" in writer.last_error

        # Recovers once the failure clears
        monkeypatch.undo()
        assert await writer.write({"response": "saved"}) is True
        assert len(read_records(writer.path)) == 1

    async def test_backup_failure_does_not_lose_the_record(self, writer, monkeypatch):
        for i in range(4):
            await writer.write({"response": f"r{i}"})

        def boom(*args, **kwargs):
            raise OSError("backup device offline")

        monkeypatch.setattr("shutil.copyfile", boom)
        assert await writer.write({"response": "r4"}) is False  # backup failed
        monkeypatch.undo()

        # The record itself still reached the transcript
        assert len(read_records(writer.path)) == 5
        assert writer.errors == 1
        assert list(writer.backup_dir.glob(".tmp-backup-*")) == []

    async def test_pause_and_resume_recording(self, writer):
        await writer.write({"response": "before"})
        writer.set_enabled(False)
        await writer.write({"response": "while paused"})
        writer.set_enabled(True)
        await writer.write({"response": "after"})

        responses = [r["response"] for r in read_records(writer.path)]
        assert responses == ["before", "after"]
        assert writer.records_skipped == 1

    async def test_torn_trailing_line_leaves_earlier_records_readable(self, writer):
        for i in range(3):
            await writer.write({"response": f"r{i}"})

        # Simulate a hard kill mid-write
        with open(writer.path, "a", encoding="utf-8") as f:
            f.write('{"seq": 4, "response": "trunca')

        good = []
        with open(writer.path, encoding="utf-8") as f:
            for line in f:
                try:
                    good.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        assert len(good) == 3

    async def test_resume_retries_setup_after_a_failed_prepare(self, tmp_path):
        target = tmp_path / "later" / "s.jsonl"
        (tmp_path / "later").write_text("blocking file")

        w = TranscriptWriter(target, backup_every=0)
        assert w.prepare() is False
        assert w.set_enabled(True) is False, "still unwritable"

        # Clear the obstruction, then resume
        (tmp_path / "later").unlink()
        assert w.set_enabled(True) is True
        assert await w.write({"response": "recovered"}) is True
        assert "recovered" in target.read_text()

    async def test_rejects_unknown_format(self, tmp_path):
        with pytest.raises(ValueError):
            TranscriptWriter(tmp_path / "s.log", fmt="csv")


class TestTextFormat:
    """The txt format stays one-record-per-line and human readable."""

    async def test_txt_records_are_single_lines(self, tmp_path):
        w = TranscriptWriter(tmp_path / "s.txt", fmt=FORMAT_TXT, backup_every=0)
        w.prepare()
        await w.write(
            {
                "response": "a person\nwaving at the camera",
                "model": "llava:7b",
                "source": "video:clip.mp4",
                "latency_ms": 412.5,
            }
        )

        lines = w.path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        assert "a person waving at the camera" in lines[0]
        assert "<video:clip.mp4>" in lines[0]
        assert "llava:7b" in lines[0]
        assert "412ms" in lines[0]

    async def test_txt_backups_work_the_same(self, tmp_path):
        w = TranscriptWriter(tmp_path / "s.txt", fmt=FORMAT_TXT, backup_every=3)
        w.prepare()
        for i in range(6):
            await w.write({"response": f"r{i}"})

        assert w.backups_written == 2
        assert len(list(w.backup_dir.glob("s.*.txt"))) == 2


class TestStats:
    """stats() reports what the UI and API need."""

    async def test_stats_shape(self, writer):
        for i in range(7):
            await writer.write({"response": f"r{i}"})

        stats = writer.stats()
        assert stats["enabled"] is True
        assert stats["ready"] is True
        assert stats["records_written"] == 7
        assert stats["backups_written"] == 1
        assert stats["records_until_backup"] == 3
        assert stats["bytes"] > 0
        assert stats["errors"] == 0
        assert stats["format"] == FORMAT_JSONL
        # Must be JSON-serializable for the API
        json.dumps(stats)

    async def test_brief_stats_are_compact(self, writer):
        await writer.write({"response": "r"})
        brief = writer.brief_stats()
        assert set(brief) == {"enabled", "records", "backups", "errors"}


class TestDefaults:
    """Default paths follow OS conventions."""

    def test_default_dir_is_absolute(self):
        assert default_transcript_dir().is_absolute()

    def test_default_name_carries_a_timestamp(self):
        name = default_transcript_name(FORMAT_JSONL, datetime(2025, 12, 17, 14, 25, 30))
        assert name == "session-20251217-142530.jsonl"

    def test_default_name_index_disambiguates(self):
        when = datetime(2025, 12, 17, 14, 25, 30)
        assert default_transcript_name(FORMAT_JSONL, when, 2) == "session-20251217-142530-2.jsonl"


class TestConcurrentInstances:
    """Several servers on one machine must never share a transcript file."""

    def test_same_second_claims_get_distinct_files(self, tmp_path):
        # Every instance generates the identical timestamped name
        when = datetime(2025, 12, 17, 14, 25, 30)
        claimed = [claim_transcript_path(tmp_path, FORMAT_JSONL, when) for _ in range(4)]

        assert len({p.name for p in claimed}) == 4, "each instance must win its own file"
        assert claimed[0].name == "session-20251217-142530.jsonl"
        assert claimed[3].name == "session-20251217-142530-3.jsonl"
        assert all(p.exists() for p in claimed)

    async def test_concurrent_writers_do_not_interleave(self, tmp_path):
        """Two writers on claimed paths keep independent, gapless sequences."""
        when = datetime(2025, 12, 17, 14, 25, 30)
        writers = []
        for _ in range(2):
            w = TranscriptWriter(
                claim_transcript_path(tmp_path, FORMAT_JSONL, when), backup_every=0
            )
            w.prepare()
            writers.append(w)

        await asyncio.gather(
            *(
                w.write({"response": f"instance {i} record {n}"})
                for i, w in enumerate(writers)
                for n in range(5)
            )
        )

        for i, w in enumerate(writers):
            records = read_records(w.path)
            assert [r["seq"] for r in records] == [1, 2, 3, 4, 5]
            assert all(f"instance {i}" in r["response"] for r in records)

    def test_claim_creates_the_directory(self, tmp_path):
        target = tmp_path / "does" / "not" / "exist"
        claimed = claim_transcript_path(target)
        assert claimed.parent == target and claimed.exists()

    def test_claim_raises_when_directory_unwritable(self, tmp_path):
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        with pytest.raises(OSError):
            claim_transcript_path(blocker / "nested")
