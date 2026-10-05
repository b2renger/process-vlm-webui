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

"""Integration tests for transcript recording: VLM wiring and the HTTP API."""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp.test_utils import AioHTTPTestCase

from process_vlm_webui import server
from process_vlm_webui.transcript import TranscriptWriter
from process_vlm_webui.vlm_service import VLMService


def fake_completion(text):
    """Build a stub matching the shape of an OpenAI chat completion."""
    message = MagicMock()
    message.content = text
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


@pytest.fixture
def image():
    from PIL import Image

    return Image.new("RGB", (32, 32), color="blue")


class TestVLMServiceRecording:
    """Every analysis produced by the VLM service reaches the transcript."""

    async def test_successful_analysis_is_recorded(self, tmp_path, image):
        writer = TranscriptWriter(tmp_path / "s.jsonl", backup_every=0)
        writer.prepare()
        svc = VLMService(model="llava:7b", transcript=writer)
        svc.client = MagicMock()
        svc.client.chat.completions.create = AsyncMock(
            return_value=fake_completion("  a blue square  ")
        )

        result = await svc.analyze_image(
            image, context={"source": "video:clip.mp4", "frame_index": 30}
        )

        assert result == "a blue square"
        records = [json.loads(line) for line in writer.path.read_text().splitlines()]
        assert len(records) == 1
        rec = records[0]
        assert rec["response"] == "a blue square"
        assert rec["model"] == "llava:7b"
        assert rec["source"] == "video:clip.mp4"
        assert rec["frame_index"] == 30
        assert rec["ok"] is True
        assert rec["latency_ms"] >= 0
        assert rec["ts"]

    async def test_failed_analysis_is_recorded(self, tmp_path, image):
        writer = TranscriptWriter(tmp_path / "s.jsonl", backup_every=0)
        writer.prepare()
        svc = VLMService(model="llava:7b", transcript=writer)
        svc.client = MagicMock()
        svc.client.chat.completions.create = AsyncMock(side_effect=RuntimeError("model offline"))

        result = await svc.analyze_image(image)

        assert result.startswith("Error:")
        rec = json.loads(writer.path.read_text().splitlines()[0])
        assert rec["ok"] is False
        assert "model offline" in rec["error"]

    async def test_every_processed_frame_is_recorded(self, tmp_path, image):
        writer = TranscriptWriter(tmp_path / "s.jsonl", backup_every=3, backup_keep=0)
        writer.prepare()
        svc = VLMService(model="llava:7b", transcript=writer)
        svc.client = MagicMock()
        svc.client.chat.completions.create = AsyncMock(
            side_effect=[fake_completion(f"frame {i}") for i in range(6)]
        )

        for i in range(6):
            await svc.analyze_frame(
                image, context={"source": "video:clip.mp4", "frame_index": i * 30}
            )

        records = [json.loads(line) for line in writer.path.read_text().splitlines()]
        assert [r["response"] for r in records] == [f"frame {i}" for i in range(6)]
        assert all(r["source"] == "video:clip.mp4" for r in records)
        assert writer.backups_written == 2

    async def test_transcript_failure_does_not_break_inference(self, tmp_path, image):
        writer = TranscriptWriter(tmp_path / "s.jsonl", backup_every=0)
        writer.prepare()
        svc = VLMService(model="llava:7b", transcript=writer)
        svc.client = MagicMock()
        svc.client.chat.completions.create = AsyncMock(return_value=fake_completion("still works"))

        with patch.object(writer, "write", side_effect=OSError("disk exploded")):
            result = await svc.analyze_image(image)

        assert result == "still works", "inference must survive a transcript failure"

    async def test_no_transcript_configured_is_a_noop(self, image):
        svc = VLMService(model="llava:7b")
        svc.client = MagicMock()
        svc.client.chat.completions.create = AsyncMock(return_value=fake_completion("no recording"))

        assert await svc.analyze_image(image) == "no recording"
        assert "transcript" not in svc.get_metrics()

    async def test_metrics_expose_transcript_state(self, tmp_path, image):
        writer = TranscriptWriter(tmp_path / "s.jsonl", backup_every=0)
        writer.prepare()
        svc = VLMService(model="llava:7b", transcript=writer)
        svc.client = MagicMock()
        svc.client.chat.completions.create = AsyncMock(return_value=fake_completion("hello"))

        await svc.analyze_image(image)

        metrics = svc.get_metrics()
        assert metrics["transcript"] == {
            "enabled": True,
            "records": 1,
            "backups": 0,
            "errors": 0,
        }


class TestTranscriptAPI(AioHTTPTestCase):
    """The /api/transcript/* endpoints."""

    async def get_application(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        writer = TranscriptWriter(
            Path(self._tmpdir.name) / "session.jsonl", backup_every=5, backup_keep=2
        )
        writer.prepare()
        server.transcript_writer = writer
        self.writer = writer
        return await server.create_app(test_mode=True)

    async def tearDownAsync(self):
        server.transcript_writer = None
        self._tmpdir.cleanup()
        await super().tearDownAsync()

    async def test_status_reports_writer_state(self):
        await self.writer.write({"response": "one"})

        resp = await self.client.get("/api/transcript/status")
        assert resp.status == 200
        data = await resp.json()
        assert data["available"] is True
        assert data["enabled"] is True
        assert data["records_written"] == 1
        assert data["filename"] == "session.jsonl"
        assert data["records_until_backup"] == 4

    async def test_recent_returns_parsed_records_newest_last(self):
        for i in range(10):
            await self.writer.write({"response": f"r{i}"})

        resp = await self.client.get("/api/transcript/recent?limit=3")
        data = await resp.json()
        assert [r["response"] for r in data["records"]] == ["r7", "r8", "r9"]

    async def test_recent_skips_a_torn_trailing_line(self):
        await self.writer.write({"response": "good"})
        with open(self.writer.path, "a", encoding="utf-8") as f:
            f.write('{"seq": 2, "resp')

        resp = await self.client.get("/api/transcript/recent")
        data = await resp.json()
        assert [r["response"] for r in data["records"]] == ["good"]

    async def test_download_returns_the_file(self):
        await self.writer.write({"response": "downloadable"})

        resp = await self.client.get("/api/transcript/download")
        assert resp.status == 200
        assert "session.jsonl" in resp.headers["Content-Disposition"]
        assert "downloadable" in (await resp.text())

    async def test_download_404_when_empty(self):
        self.writer.path.unlink()
        resp = await self.client.get("/api/transcript/download")
        assert resp.status == 404

    async def test_backup_endpoint_creates_a_snapshot(self):
        await self.writer.write({"response": "snapshot me"})

        resp = await self.client.post("/api/transcript/backup")
        assert resp.status == 200
        data = await resp.json()
        assert data["status"] == "ok"
        assert Path(data["backup"]).exists()
        assert data["backups_written"] == 1

    async def test_backup_endpoint_skips_when_nothing_written(self):
        resp = await self.client.post("/api/transcript/backup")
        data = await resp.json()
        assert data["status"] == "skipped"

    async def test_config_toggles_recording(self):
        resp = await self.client.post("/api/transcript/config", json={"enabled": False})
        assert resp.status == 200
        assert (await resp.json())["enabled"] is False

        await self.writer.write({"response": "should be dropped"})
        assert self.writer.records_skipped == 1

        resp = await self.client.post("/api/transcript/config", json={"enabled": True})
        assert (await resp.json())["enabled"] is True
        await self.writer.write({"response": "kept"})
        assert "kept" in self.writer.path.read_text()

    async def test_config_updates_backup_interval(self):
        resp = await self.client.post("/api/transcript/config", json={"backup_every": 100})
        assert (await resp.json())["backup_every"] == 100

    async def test_config_rejects_out_of_range_interval(self):
        resp = await self.client.post("/api/transcript/config", json={"backup_every": -1})
        assert resp.status == 400

    async def test_config_rejects_non_integer_interval(self):
        resp = await self.client.post("/api/transcript/config", json={"backup_every": "lots"})
        assert resp.status == 400


class TestShutdownGuard(AioHTTPTestCase):
    """New jobs and uploads are refused once shutdown has begun."""

    async def get_application(self):
        server.transcript_writer = None
        return await server.create_app(test_mode=True)

    async def tearDownAsync(self):
        server.shutting_down = False
        await super().tearDownAsync()

    async def test_start_refused_during_shutdown(self):
        server.shutting_down = True
        resp = await self.client.post("/api/video/start", json={"video_id": "whatever"})
        assert resp.status == 503
        assert "shutting down" in (await resp.json())["error"].lower()

    async def test_start_not_refused_for_shutdown_normally(self):
        server.shutting_down = False
        # The request still fails (no model, no such video) but not on the
        # shutdown guard - that distinction is what this test pins down
        resp = await self.client.post("/api/video/start", json={"video_id": "whatever"})
        assert "shutting down" not in (await resp.json())["error"].lower()


class TestTranscriptAPIDisabled(AioHTTPTestCase):
    """Endpoints degrade gracefully when recording was never configured."""

    async def get_application(self):
        server.transcript_writer = None
        return await server.create_app(test_mode=True)

    async def test_status_reports_unavailable(self):
        resp = await self.client.get("/api/transcript/status")
        assert resp.status == 200
        assert (await resp.json())["available"] is False

    async def test_backup_returns_404(self):
        resp = await self.client.post("/api/transcript/backup")
        assert resp.status == 404
