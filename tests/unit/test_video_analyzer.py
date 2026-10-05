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

"""Unit tests for video probing, sample planning and the analysis job."""

import asyncio
import json

import pytest

from process_vlm_webui.transcript import TranscriptWriter
from process_vlm_webui.video_analyzer import (
    STEP_FRAMES,
    STEP_SECONDS,
    VideoAnalysisJob,
    format_timecode,
    plan_sample_indices,
    probe_video,
)


class FakeVLMService:
    """Stands in for VLMService, recording what it was asked to describe."""

    def __init__(self, transcript=None, responses=None, fail_on=()):
        self.transcript = transcript
        self.prompt = "Describe this frame."
        self.model = "fake-vlm"
        self.api_base = "http://localhost/v1"
        self.calls = []
        self.responses = responses
        self.fail_on = set(fail_on)
        self.last_inference_time = 0.01
        self.delay = 0.0

    async def analyze_frame(self, image, prompt=None, context=None):
        if self.delay:
            await asyncio.sleep(self.delay)
        index = len(self.calls)
        self.calls.append({"prompt": prompt, "context": context, "size": image.size})

        if index in self.fail_on:
            text = "Error: model offline"
        elif self.responses:
            text = self.responses[index % len(self.responses)]
        else:
            text = f"description {index}"

        if self.transcript:
            await self.transcript.write(
                {"model": self.model, "prompt": prompt, "response": text, **(context or {})}
            )
        return text

    def get_metrics(self):
        return {"total_inferences": len(self.calls)}


# ----------------------------------------------------------------------
# Timecode
# ----------------------------------------------------------------------


class TestFormatTimecode:
    def test_zero(self):
        assert format_timecode(0) == "00:00:00.000"

    def test_sub_second_precision_is_kept(self):
        assert format_timecode(1.25) == "00:00:01.250"

    def test_hours_minutes_seconds(self):
        assert format_timecode(3661.5) == "01:01:01.500"

    def test_negative_and_none_are_clamped(self):
        assert format_timecode(-5) == "00:00:00.000"
        assert format_timecode(None) == "00:00:00.000"


# ----------------------------------------------------------------------
# Probing
# ----------------------------------------------------------------------


class TestProbeVideo:
    def test_reads_real_metadata(self, tiny_video):
        path, fps, frames = tiny_video
        meta = probe_video(path)

        assert meta["filename"] == "tiny.mp4"
        assert meta["fps"] == pytest.approx(fps, abs=0.1)
        assert meta["frame_count"] == frames
        assert meta["width"] == 160
        assert meta["height"] == 120
        assert meta["duration_s"] == pytest.approx(frames / fps, abs=0.1)
        assert meta["duration_timecode"].startswith("00:00:05")

    def test_unreadable_file_is_rejected(self, tmp_path):
        bogus = tmp_path / "not-a-video.mp4"
        bogus.write_bytes(b"this is not an mp4")

        with pytest.raises(ValueError, match="Cannot open"):
            probe_video(bogus)

    def test_missing_file_is_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            probe_video(tmp_path / "absent.mp4")


# ----------------------------------------------------------------------
# Sample planning
# ----------------------------------------------------------------------


class TestPlanSampleIndices:
    meta = {"fps": 30.0, "frame_count": 900, "duration_s": 30.0}

    def test_time_step_becomes_a_frame_stride(self):
        indices = plan_sample_indices(self.meta, step_seconds=2.0)
        assert indices[:4] == [0, 60, 120, 180]
        assert len(indices) == 15

    def test_sub_second_step(self):
        indices = plan_sample_indices(self.meta, step_seconds=0.5)
        assert indices[:3] == [0, 15, 30]

    def test_step_shorter_than_a_frame_still_advances(self):
        # Never return the same frame twice - stride is clamped to 1
        indices = plan_sample_indices(self.meta, step_seconds=0.001)
        assert indices[:3] == [0, 1, 2]

    def test_frame_step_mode(self):
        indices = plan_sample_indices(self.meta, step_mode=STEP_FRAMES, step_frames=100)
        assert indices == [0, 100, 200, 300, 400, 500, 600, 700, 800]

    def test_range_is_honoured(self):
        indices = plan_sample_indices(self.meta, step_seconds=1.0, start_s=5.0, end_s=10.0)
        assert indices[0] == 150
        assert indices[-1] <= 300

    def test_max_frames_caps_the_plan(self):
        indices = plan_sample_indices(self.meta, step_seconds=0.1, max_frames=7)
        assert len(indices) == 7

    def test_unknown_frame_count_falls_back_to_duration(self):
        meta = {"fps": 30.0, "frame_count": 0, "duration_s": 10.0}
        indices = plan_sample_indices(meta, step_seconds=2.0)
        assert indices == [0, 60, 120, 180, 240]

    def test_zero_step_is_rejected(self):
        with pytest.raises(ValueError, match="step_seconds"):
            plan_sample_indices(self.meta, step_seconds=0)

    def test_bad_step_mode_is_rejected(self):
        with pytest.raises(ValueError, match="step mode"):
            plan_sample_indices(self.meta, step_mode="vibes")


# ----------------------------------------------------------------------
# The job itself
# ----------------------------------------------------------------------


@pytest.mark.asyncio
class TestVideoAnalysisJob:
    async def test_describes_every_planned_frame_in_order(self, tiny_video, tmp_path):
        path, fps, frames = tiny_video
        vlm = FakeVLMService()

        job = VideoAnalysisJob(
            path, vlm, step_mode=STEP_SECONDS, step_seconds=1.0, frames_dir=tmp_path / "frames"
        )
        assert len(job.indices) == 5  # 5 seconds at one per second

        await job.start()._task

        assert job.state == "completed"
        assert job.analyzed == 5
        assert [r["step"] for r in job.results] == [1, 2, 3, 4, 5]
        # Timecodes advance monotonically through the video
        times = [r["video_time_s"] for r in job.results]
        assert times == sorted(times)
        assert times[0] == 0.0
        assert times[1] == pytest.approx(1.0, abs=0.05)

    async def test_frames_are_downscaled_before_inference(self, tiny_video, tmp_path):
        path, _, _ = tiny_video
        vlm = FakeVLMService()

        job = VideoAnalysisJob(
            path, vlm, step_seconds=2.0, max_width=80, frames_dir=tmp_path / "frames"
        )
        await job.start()._task

        # Source is 160x120, so a max width of 80 halves both dimensions
        assert vlm.calls[0]["size"] == (80, 60)

    async def test_every_analysis_reaches_the_transcript(self, tiny_video, tmp_path):
        path, _, _ = tiny_video
        writer = TranscriptWriter(tmp_path / "session.jsonl", backup_every=0)
        writer.prepare()
        vlm = FakeVLMService(transcript=writer)

        job = VideoAnalysisJob(path, vlm, step_seconds=1.0, frames_dir=tmp_path / "frames")
        await job.start()._task

        records = [json.loads(line) for line in writer.path.read_text().splitlines()]
        kinds = [r.get("kind") for r in records]
        assert kinds[0] == "video_job_start"
        assert kinds[-1] == "video_job_end"

        analyses = [r for r in records if r.get("kind") == "analysis"]
        assert len(analyses) == 5
        # Each analysis carries where in the video it came from
        assert all("timecode" in r and "video_time_s" in r for r in analyses)
        assert all(r["video"] == "tiny.mp4" for r in analyses)
        assert [r["step"] for r in analyses] == [1, 2, 3, 4, 5]
        assert records[-1]["analyzed"] == 5
        assert records[-1]["state"] == "completed"

    async def test_failed_inferences_are_counted_not_fatal(self, tiny_video, tmp_path):
        path, _, _ = tiny_video
        vlm = FakeVLMService(fail_on=(1, 3))

        job = VideoAnalysisJob(path, vlm, step_seconds=1.0, frames_dir=tmp_path / "frames")
        await job.start()._task

        assert job.state == "completed"
        assert job.analyzed == 3
        assert job.failed == 2
        assert [r["ok"] for r in job.results] == [True, False, True, False, True]

    async def test_frame_images_are_written_and_linked(self, tiny_video, tmp_path):
        path, _, _ = tiny_video
        frames_dir = tmp_path / "frames"

        job = VideoAnalysisJob(
            path, FakeVLMService(), step_seconds=2.0, frames_dir=frames_dir, save_frames=True
        )
        await job.start()._task

        saved = sorted((frames_dir / job.job_id).glob("*.jpg"))
        assert len(saved) == len(job.results)
        assert job.results[0]["frame_url"] == f"/api/video/frame/{job.job_id}/{saved[0].name}"

    async def test_frames_can_be_disabled(self, tiny_video, tmp_path):
        path, _, _ = tiny_video
        job = VideoAnalysisJob(
            path,
            FakeVLMService(),
            step_seconds=2.0,
            frames_dir=tmp_path / "frames",
            save_frames=False,
        )
        await job.start()._task

        assert not (tmp_path / "frames" / job.job_id).exists()
        assert all(r["frame_url"] is None for r in job.results)

    async def test_carry_context_quotes_the_previous_description(self, tiny_video, tmp_path):
        path, _, _ = tiny_video
        vlm = FakeVLMService(responses=["a cat", "a dog"])

        job = VideoAnalysisJob(
            path,
            vlm,
            step_seconds=1.0,
            carry_context=True,
            frames_dir=tmp_path / "frames",
        )
        await job.start()._task

        # The first prompt has nothing to carry; later ones quote the last answer
        assert "previous sampled moment" not in vlm.calls[0]["prompt"]
        assert '"a cat"' in vlm.calls[1]["prompt"]
        assert '"a dog"' in vlm.calls[2]["prompt"]

    async def test_context_is_off_by_default(self, tiny_video, tmp_path):
        path, _, _ = tiny_video
        vlm = FakeVLMService()

        job = VideoAnalysisJob(path, vlm, step_seconds=1.0, frames_dir=tmp_path / "frames")
        await job.start()._task

        assert all("previous sampled moment" not in c["prompt"] for c in vlm.calls)

    async def test_cancel_stops_early_and_reports_it(self, tiny_video, tmp_path):
        path, _, _ = tiny_video
        vlm = FakeVLMService()
        vlm.delay = 0.05

        job = VideoAnalysisJob(
            path, vlm, step_mode=STEP_FRAMES, step_frames=1, frames_dir=tmp_path / "frames"
        )
        job.start()
        await asyncio.sleep(0.12)
        await job.cancel()

        assert job.state == "cancelled"
        assert 0 < job.analyzed < len(job.indices)

    async def test_pause_holds_then_resumes(self, tiny_video, tmp_path):
        path, _, _ = tiny_video
        vlm = FakeVLMService()
        vlm.delay = 0.02

        job = VideoAnalysisJob(
            path, vlm, step_mode=STEP_FRAMES, step_frames=1, frames_dir=tmp_path / "frames"
        )
        job.start()
        await asyncio.sleep(0.05)
        job.pause()
        assert job.state == "paused"

        paused_at = job.analyzed
        await asyncio.sleep(0.1)
        # At most the one frame that was already in flight may have landed
        assert job.analyzed - paused_at <= 1

        job.resume()
        assert job.state == "running"
        await job._task
        assert job.state == "completed"

    async def test_events_are_emitted_for_the_ui(self, tiny_video, tmp_path):
        path, _, _ = tiny_video
        events = []

        job = VideoAnalysisJob(
            path,
            FakeVLMService(),
            step_seconds=2.0,
            frames_dir=tmp_path / "frames",
            on_event=events.append,
        )
        await job.start()._task

        kinds = [e["type"] for e in events]
        assert kinds[0] == "job_started"
        assert kinds[-1] == "job_finished"
        assert kinds.count("video_analysis") == len(job.results)

    async def test_a_broken_event_callback_cannot_stop_the_job(self, tiny_video, tmp_path):
        path, _, _ = tiny_video

        def explode(_event):
            raise RuntimeError("the UI broadcaster is on fire")

        job = VideoAnalysisJob(
            path,
            FakeVLMService(),
            step_seconds=2.0,
            frames_dir=tmp_path / "frames",
            on_event=explode,
        )
        await job.start()._task

        assert job.state == "completed"
        assert job.analyzed == len(job.indices)

    async def test_status_reports_progress(self, tiny_video, tmp_path):
        path, _, _ = tiny_video

        job = VideoAnalysisJob(
            path, FakeVLMService(), step_seconds=1.0, frames_dir=tmp_path / "frames"
        )
        before = job.status()
        assert before["state"] == "queued"
        assert before["progress"] == 0.0
        assert before["total"] == 5

        await job.start()._task

        after = job.status()
        assert after["state"] == "completed"
        assert after["progress"] == 1.0
        assert after["active"] is False
        assert after["elapsed_s"] >= 0

    async def test_range_limits_which_part_is_described(self, tiny_video, tmp_path):
        path, _, _ = tiny_video

        job = VideoAnalysisJob(
            path,
            FakeVLMService(),
            step_seconds=1.0,
            start_s=2.0,
            end_s=4.0,
            frames_dir=tmp_path / "frames",
        )
        await job.start()._task

        times = [r["video_time_s"] for r in job.results]
        assert times[0] == pytest.approx(2.0, abs=0.05)
        assert max(times) <= 4.05

    async def test_starting_twice_is_refused(self, tiny_video, tmp_path):
        path, _, _ = tiny_video
        job = VideoAnalysisJob(
            path, FakeVLMService(), step_seconds=2.0, frames_dir=tmp_path / "frames"
        )
        job.start()
        with pytest.raises(RuntimeError, match="already been started"):
            job.start()
        await job.cancel()
