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

"""Integration tests for the /api/video endpoints."""

import asyncio
import tempfile
from pathlib import Path

import cv2
import numpy as np
from aiohttp import FormData
from aiohttp.test_utils import AioHTTPTestCase

from process_vlm_webui import server


def write_tiny_video(path, fps=10, frames=30, size=(160, 120)):
    """Write a small real mp4 so the server has something decodable to probe."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert writer.isOpened(), "OpenCV could not open an mp4 writer"
    for i in range(frames):
        writer.write(np.full((size[1], size[0], 3), i * 8 % 256, dtype=np.uint8))
    writer.release()
    return path


class StubVLM:
    """Minimal stand-in for VLMService so no model is needed."""

    def __init__(self):
        self.model = "stub-vlm"
        self.api_base = "http://localhost/v1"
        self.prompt = "Describe this frame."
        self.transcript = None
        self.last_inference_time = 0.001
        self.calls = 0
        self.updated_prompt = None

    async def analyze_frame(self, image, prompt=None, context=None):
        self.calls += 1
        return f"stub description {self.calls}"

    def get_metrics(self):
        return {"total_inferences": self.calls}

    def update_prompt(self, prompt, max_tokens=None):
        self.updated_prompt = prompt
        self.prompt = prompt


class VideoAPITestCase(AioHTTPTestCase):
    """Shared setup: a temp storage root, a stub model and one real video file."""

    async def get_application(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)

        server.transcript_writer = None
        server.vlm_service = StubVLM()
        server.videos.clear()
        server.jobs.clear()
        server.current_job = None
        server.shutting_down = False
        server.configure_storage(root / "videos", root / "frames", max_upload_mb=64)

        self.source_video = write_tiny_video(root / "source.mp4")
        return await server.create_app(test_mode=True)

    async def tearDownAsync(self):
        if server.current_job is not None:
            await server.current_job.cancel()
        server.current_job = None
        server.videos.clear()
        server.jobs.clear()
        server.allow_server_paths = False
        await super().tearDownAsync()
        self._tmp.cleanup()

    async def upload(self, path=None, filename="clip.mp4"):
        """Upload a video through the real multipart endpoint."""
        path = path or self.source_video
        form = FormData()
        with open(path, "rb") as f:
            form.add_field("file", f.read(), filename=filename, content_type="video/mp4")
            return await self.client.post("/api/video/upload", data=form)

    async def run_job(self, video_id, **options):
        """Start a job and wait for it to reach a terminal state."""
        body = {"video_id": video_id, "step_seconds": 1.0}
        body.update(options)
        resp = await self.client.post("/api/video/start", json=body)
        assert resp.status == 200, await resp.text()
        job_id = (await resp.json())["job"]["job_id"]

        for _ in range(200):
            if not server.jobs[job_id].status()["active"]:
                break
            await asyncio.sleep(0.02)
        return server.jobs[job_id]


class TestUpload(VideoAPITestCase):
    async def test_upload_stores_and_probes_the_file(self):
        resp = await self.upload()
        assert resp.status == 200

        video = (await resp.json())["video"]
        assert video["original_filename"] == "clip.mp4"
        assert video["frame_count"] == 30
        assert video["width"] == 160
        assert video["duration_s"] > 0
        assert Path(video["path"]).exists()

    async def test_uploaded_video_is_listed_and_playable(self):
        video = (await (await self.upload()).json())["video"]

        listing = await (await self.client.get("/api/video/list")).json()
        assert listing["count"] == 1
        assert listing["videos"][0]["video_id"] == video["video_id"]

        playback = await self.client.get(f"/api/video/file/{video['video_id']}")
        assert playback.status == 200
        assert len(await playback.read()) > 0

    async def test_non_video_extension_is_refused(self):
        resp = await self.upload(filename="notes.txt")
        assert resp.status == 400
        assert "Unsupported" in (await resp.json())["error"]

    async def test_undecodable_file_is_refused_and_not_kept(
        self,
    ):
        junk = Path(self._tmp.name) / "junk.mp4"
        junk.write_bytes(b"definitely not video data")

        resp = await self.upload(path=junk)
        assert resp.status == 400
        assert list(server.videos) == []
        # The partial upload is cleaned up rather than left behind
        assert list(server.videos_dir.glob("*.mp4")) == []

    async def test_missing_file_part_is_refused(self):
        resp = await self.client.post("/api/video/upload", data={"other": "value"})
        assert resp.status == 400

    async def test_unknown_video_id_is_404(self):
        resp = await self.client.get("/api/video/file/does-not-exist")
        assert resp.status == 404


class TestJobLifecycle(VideoAPITestCase):
    async def test_start_analyses_the_whole_video(self):
        video = (await (await self.upload()).json())["video"]
        job = await self.run_job(video["video_id"])

        assert job.state == "completed"
        # 30 frames at 10fps = 3s, sampled once per second
        assert job.analyzed == 3
        assert [r["timecode"] for r in job.results][0] == "00:00:00.000"

    async def test_results_endpoint_returns_the_descriptions(self):
        video = (await (await self.upload()).json())["video"]
        job = await self.run_job(video["video_id"])

        data = await (await self.client.get(f"/api/video/results?job_id={job.job_id}")).json()
        assert data["total"] == 3
        assert data["results"][0]["text"] == "stub description 1"
        assert data["job"]["state"] == "completed"

    async def test_results_paginate(self):
        video = (await (await self.upload()).json())["video"]
        job = await self.run_job(video["video_id"])

        data = await (
            await self.client.get(f"/api/video/results?job_id={job.job_id}&offset=1&limit=1")
        ).json()
        assert data["count"] == 1
        assert data["total"] == 3
        assert data["results"][0]["step"] == 2

    async def test_status_tracks_the_current_job(self):
        video = (await (await self.upload()).json())["video"]

        idle = await (await self.client.get("/api/video/status")).json()
        assert idle["active"] is False

        job = await self.run_job(video["video_id"])
        done = await (await self.client.get("/api/video/status")).json()
        assert done["job"]["job_id"] == job.job_id
        assert done["job"]["progress"] == 1.0

    async def test_frame_images_are_served(self):
        video = (await (await self.upload()).json())["video"]
        job = await self.run_job(video["video_id"])

        url = job.results[0]["frame_url"]
        resp = await self.client.get(url)
        assert resp.status == 200
        assert (await resp.read())[:2] == b"\xff\xd8"  # JPEG magic

    async def test_frame_path_traversal_is_refused(self):
        resp = await self.client.get("/api/video/frame/..%2F..%2Fetc/passwd")
        assert resp.status in (400, 404)

    async def test_prompt_override_reaches_the_model(self):
        video = (await (await self.upload()).json())["video"]
        await self.run_job(video["video_id"], prompt="Count the people.")

        assert server.vlm_service.updated_prompt == "Count the people."

    async def test_second_job_is_refused_while_one_runs(self):
        video = (await (await self.upload()).json())["video"]

        first = await self.client.post(
            "/api/video/start",
            json={"video_id": video["video_id"], "step_frames": 1, "step_mode": "frames"},
        )
        assert first.status == 200

        second = await self.client.post(
            "/api/video/start", json={"video_id": video["video_id"], "step_seconds": 1.0}
        )
        assert second.status == 409

        await self.client.post("/api/video/stop", json={})

    async def test_stop_cancels_the_running_job(self):
        video = (await (await self.upload()).json())["video"]

        await self.client.post(
            "/api/video/start",
            json={"video_id": video["video_id"], "step_mode": "frames", "step_frames": 1},
        )
        resp = await self.client.post("/api/video/stop", json={})
        assert resp.status == 200
        assert (await resp.json())["job"]["state"] == "cancelled"

    async def test_pause_and_resume(self):
        video = (await (await self.upload()).json())["video"]
        await self.client.post(
            "/api/video/start",
            json={"video_id": video["video_id"], "step_mode": "frames", "step_frames": 1},
        )

        paused = await self.client.post("/api/video/pause", json={"paused": True})
        assert (await paused.json())["job"]["state"] in ("paused", "completed")

        resumed = await self.client.post("/api/video/pause", json={"paused": False})
        assert (await resumed.json())["job"]["state"] in ("running", "completed")

        await self.client.post("/api/video/stop", json={})

    async def test_start_with_unknown_video_is_404(self):
        resp = await self.client.post("/api/video/start", json={"video_id": "nope"})
        assert resp.status == 404

    async def test_step_selecting_no_frames_is_rejected(self):
        video = (await (await self.upload()).json())["video"]
        resp = await self.client.post(
            "/api/video/start", json={"video_id": video["video_id"], "start_s": 9999}
        )
        assert resp.status == 400
        assert "no frames" in (await resp.json())["error"]

    async def test_invalid_step_is_rejected(self):
        video = (await (await self.upload()).json())["video"]
        resp = await self.client.post(
            "/api/video/start", json={"video_id": video["video_id"], "step_seconds": 0}
        )
        assert resp.status == 400


class TestServerPaths(VideoAPITestCase):
    async def test_register_is_refused_by_default(self):
        resp = await self.client.post("/api/video/register", json={"path": str(self.source_video)})
        assert resp.status == 403

    async def test_register_works_when_enabled(self):
        server.allow_server_paths = True
        resp = await self.client.post("/api/video/register", json={"path": str(self.source_video)})
        assert resp.status == 200
        assert (await resp.json())["video"]["frame_count"] == 30

    async def test_register_rejects_a_missing_path(self):
        server.allow_server_paths = True
        resp = await self.client.post("/api/video/register", json={"path": "/no/such/clip.mp4"})
        assert resp.status == 404


class TestDelete(VideoAPITestCase):
    async def test_delete_removes_an_uploaded_file(self):
        video = (await (await self.upload()).json())["video"]
        path = Path(video["path"])
        assert path.exists()

        resp = await self.client.post("/api/video/delete", json={"video_id": video["video_id"]})
        assert resp.status == 200
        assert (await resp.json())["file_removed"] is True
        assert not path.exists()
        assert video["video_id"] not in server.videos

    async def test_delete_leaves_server_side_files_alone(self):
        server.allow_server_paths = True
        registered = await (
            await self.client.post("/api/video/register", json={"path": str(self.source_video)})
        ).json()

        resp = await self.client.post(
            "/api/video/delete", json={"video_id": registered["video"]["video_id"]}
        )
        assert (await resp.json())["file_removed"] is False
        assert self.source_video.exists()

    async def test_delete_unknown_video_is_404(self):
        resp = await self.client.post("/api/video/delete", json={"video_id": "nope"})
        assert resp.status == 404


class TestReporting(VideoAPITestCase):
    async def test_the_users_filename_is_what_gets_reported(self):
        """Uploads are stored under a generated id, but reports must not be."""
        video = (await (await self.upload(filename="site-survey.mp4")).json())["video"]
        job = await self.run_job(video["video_id"])

        assert job.meta["filename"] == "site-survey.mp4"
        assert job.status()["video"] == "site-survey.mp4"
        # The generated name is kept too, so the file on disk is still findable
        assert job.meta["stored_filename"].endswith(".mp4")
        assert job.meta["stored_filename"] != "site-survey.mp4"


class TestModelListing(VideoAPITestCase):
    """The running model must come back flagged, or the UI replaces it."""

    async def get_application(self):
        app = await super().get_application()
        server.vlm_service.model = "gemma4:12b"
        server.vlm_service.api_base = "http://localhost:11434/v1"
        return app

    async def _patch_client(self, ids):
        from unittest.mock import AsyncMock, MagicMock

        listing = MagicMock()
        listing.data = [MagicMock(id=i) for i in ids]
        client = MagicMock()
        client.models.list = AsyncMock(return_value=listing)
        server.vlm_service.client = client

    async def test_running_model_is_flagged_for_the_same_endpoint(self):
        await self._patch_client(["qwen3.8:latest", "gemma4:12b"])

        resp = await self.client.get("/models?api_base=http://localhost:11434/v1")
        models = (await resp.json())["models"]

        assert [m["current"] for m in models] == [False, True]

    async def test_a_trailing_slash_still_matches(self):
        await self._patch_client(["gemma4:12b"])

        resp = await self.client.get("/models?api_base=http://localhost:11434/v1/")
        assert (await resp.json())["models"][0]["current"] is True

    async def test_a_different_endpoint_is_queried_separately(self):
        """Asking about another backend must not report this one's models."""
        await self._patch_client(["gemma4:12b"])

        await self.client.get("/models?api_base=http://localhost:19999/v1")

        # The live client belongs to the configured endpoint only
        server.vlm_service.client.models.list.assert_not_called()
