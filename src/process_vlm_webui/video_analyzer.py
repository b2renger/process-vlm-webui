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
Video Analyzer
Samples an MP4 (or any OpenCV-readable video) at a configurable time step and
sends every sampled frame to the VLM, exactly one at a time.

Design goals:
  * Deterministic, ordered output - frames are analysed strictly in time order
    and one at a time, so the transcript reads as a description of the video
    from start to finish with no interleaving.
  * Never block the event loop - all OpenCV decoding happens in a worker
    thread; only the VLM call and the progress broadcast run on the loop.
  * Resumable control - a job can be paused, resumed and cancelled while it
    runs, and always reports a final status.
  * Same logging as the live app - every analysis goes through
    VLMService.analyze_frame(), so the TranscriptWriter records it in the
    identical format, enriched with the video timecode it came from.
"""

import asyncio
import logging
import math
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
from PIL import Image

from .vlm_service import VLMService

logger = logging.getLogger(__name__)

# Sampling modes
STEP_SECONDS = "seconds"
STEP_FRAMES = "frames"
STEP_MODES = (STEP_SECONDS, STEP_FRAMES)

# Container formats we accept on upload
VIDEO_SUFFIXES = (".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".mpg", ".mpeg", ".wmv")

# Beyond this many frames to skip, a random seek beats decoding every frame
SEEK_THRESHOLD_FRAMES = 60

# Fallback when a container reports a nonsensical frame rate
FALLBACK_FPS = 30.0

# Job lifecycle states
STATE_QUEUED = "queued"
STATE_RUNNING = "running"
STATE_PAUSED = "paused"
STATE_DONE = "completed"
STATE_CANCELLED = "cancelled"
STATE_FAILED = "failed"


def format_timecode(seconds: float) -> str:
    """Render a position in the video as HH:MM:SS.mmm"""
    if seconds is None or seconds < 0 or math.isnan(seconds):
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def probe_video(path: Path | str) -> dict[str, Any]:
    """
    Read a video's metadata without decoding it.

    Raises ValueError when the file cannot be opened, so the upload handler can
    reject an unusable file before a job is ever created.
    """
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {path.name}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
    finally:
        cap.release()

    # Some containers report 0 or absurd rates; a sane default keeps the time
    # step meaningful instead of producing a division by zero downstream
    if not fps or fps <= 0 or fps > 1000:
        logger.warning(f"Implausible FPS ({fps}) in {path.name}; assuming {FALLBACK_FPS}")
        fps = FALLBACK_FPS

    codec = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)).strip() if fourcc else ""
    duration = frame_count / fps if frame_count > 0 else 0.0

    return {
        "filename": path.name,
        "path": str(path),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "fps": round(fps, 3),
        "frame_count": frame_count,
        "duration_s": round(duration, 3),
        "duration_timecode": format_timecode(duration),
        "width": width,
        "height": height,
        "codec": codec,
    }


def plan_sample_indices(
    meta: dict[str, Any],
    step_mode: str = STEP_SECONDS,
    step_seconds: float = 2.0,
    step_frames: int = 30,
    start_s: float = 0.0,
    end_s: Optional[float] = None,
    max_frames: int = 0,
) -> list[int]:
    """
    Work out which frame indices to analyse, before any decoding happens.

    Returning the plan up front is what lets the UI show a total and an ETA
    from the first moment, and makes the sampling trivially testable.
    """
    fps = float(meta.get("fps") or FALLBACK_FPS)
    frame_count = int(meta.get("frame_count") or 0)
    duration = float(meta.get("duration_s") or 0.0)

    if step_mode not in STEP_MODES:
        raise ValueError(f"Unsupported step mode: {step_mode!r} (expected one of {STEP_MODES})")

    if step_mode == STEP_SECONDS:
        if step_seconds <= 0:
            raise ValueError("step_seconds must be greater than 0")
        stride = max(1, int(round(step_seconds * fps)))
    else:
        if step_frames < 1:
            raise ValueError("step_frames must be at least 1")
        stride = int(step_frames)

    first = max(0, int(round(max(0.0, start_s) * fps)))

    # An unknown frame count (streamed/truncated container) still has to end
    # somewhere; fall back to the duration, then to the first frame only.
    last = frame_count - 1 if frame_count > 0 else int(round(duration * fps)) - 1
    if end_s is not None and end_s > 0:
        last = min(last, int(round(end_s * fps)))
    if last < first:
        return [first] if first == 0 else []

    indices = list(range(first, last + 1, stride))
    if max_frames and max_frames > 0:
        indices = indices[:max_frames]
    return indices


def _resize_for_vlm(frame, max_width: int):
    """Downscale a decoded BGR frame so the VLM sees fewer, cheaper pixels."""
    if not max_width or max_width <= 0:
        return frame
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / float(width)
    return cv2.resize(
        frame, (max_width, max(1, int(round(height * scale)))), interpolation=cv2.INTER_AREA
    )


class VideoAnalysisJob:
    """
    One video, sampled at a fixed step and described frame by frame.

    The job owns its own VideoCapture and drives it from a worker thread, so a
    slow decode never stalls the WebSocket broadcasts or the HTTP API.
    """

    def __init__(
        self,
        video_path: Path | str,
        vlm_service: VLMService,
        *,
        job_id: Optional[str] = None,
        video_id: Optional[str] = None,
        display_name: Optional[str] = None,
        step_mode: str = STEP_SECONDS,
        step_seconds: float = 2.0,
        step_frames: int = 30,
        start_s: float = 0.0,
        end_s: Optional[float] = None,
        max_frames: int = 0,
        max_width: int = 1280,
        prompt: Optional[str] = None,
        carry_context: bool = False,
        save_frames: bool = True,
        frames_dir: Optional[Path] = None,
        jpeg_quality: int = 85,
        on_event: Optional[Callable[[dict], None]] = None,
    ):
        self.job_id = job_id or uuid.uuid4().hex[:12]
        self.video_path = Path(video_path)
        self.video_id = video_id or self.video_path.stem
        self.vlm_service = vlm_service

        self.step_mode = step_mode
        self.step_seconds = float(step_seconds)
        self.step_frames = int(step_frames)
        self.start_s = float(start_s)
        self.end_s = end_s
        self.max_frames = int(max_frames)
        self.max_width = int(max_width)
        self.prompt = prompt
        self.carry_context = bool(carry_context)
        self.jpeg_quality = int(jpeg_quality)
        self.on_event = on_event

        self.meta = probe_video(self.video_path)
        # Uploads are stored under a generated id, but every report - the UI,
        # the log line, the transcript - should name the file the user chose
        if display_name:
            self.meta["stored_filename"] = self.meta["filename"]
            self.meta["filename"] = display_name
        self.indices = plan_sample_indices(
            self.meta,
            step_mode=step_mode,
            step_seconds=step_seconds,
            step_frames=step_frames,
            start_s=start_s,
            end_s=end_s,
            max_frames=max_frames,
        )

        self.save_frames = bool(save_frames)
        self.frames_dir = Path(frames_dir) / self.job_id if frames_dir else None
        if self.save_frames and self.frames_dir:
            try:
                self.frames_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                # Thumbnails are a convenience; losing them must not stop the run
                logger.warning(f"Cannot create frame directory {self.frames_dir}: {e}")
                self.save_frames = False

        # Runtime state
        self.state = STATE_QUEUED
        self.analyzed = 0
        self.failed = 0
        self.error: Optional[str] = None
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self.current_index: Optional[int] = None
        self.current_time_s: Optional[float] = None
        self.results: list[dict[str, Any]] = []
        self.last_response: Optional[str] = None

        self._task: Optional[asyncio.Task] = None
        self._cancelled = False
        self._resume = asyncio.Event()
        self._resume.set()

        # Every VideoCapture call runs on this one thread. Cancelling the job
        # only interrupts the *await*, never the decode already running inside
        # OpenCV - releasing the capture from another thread while that decode
        # is in flight crashes the interpreter. A single worker serializes
        # open, read and release, so the release always queues behind the read.
        self._capture_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"vlm-video-{self.job_id}"
        )

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def start(self) -> "VideoAnalysisJob":
        """Launch the analysis in the background and return immediately."""
        if self._task is not None:
            raise RuntimeError(f"Job {self.job_id} has already been started")
        self._task = asyncio.create_task(self._run())
        return self

    def pause(self) -> None:
        """Hold after the in-flight frame finishes."""
        if self.state == STATE_RUNNING:
            self._resume.clear()
            self.state = STATE_PAUSED
            self._emit({"type": "job_paused", **self.status()})

    def resume(self) -> None:
        """Continue a paused job."""
        if self.state == STATE_PAUSED:
            self.state = STATE_RUNNING
            self._resume.set()
            self._emit({"type": "job_resumed", **self.status()})

    async def cancel(self) -> None:
        """Stop the job and wait for it to unwind."""
        self._cancelled = True
        # A paused job is parked on the event; release it so it can observe
        # the cancel flag and exit instead of hanging until resume
        self._resume.set()
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """JSON-serializable snapshot of the job, for the API and the UI."""
        total = len(self.indices)
        elapsed = (self.finished_at or time.time()) - self.started_at if self.started_at else 0.0
        rate = self.analyzed / elapsed if elapsed > 0 and self.analyzed else 0.0
        remaining = max(0, total - self.analyzed)

        return {
            "job_id": self.job_id,
            "video_id": self.video_id,
            "video": self.meta.get("filename"),
            "state": self.state,
            "active": self.state in (STATE_RUNNING, STATE_PAUSED, STATE_QUEUED),
            "analyzed": self.analyzed,
            "failed": self.failed,
            "total": total,
            "progress": round(self.analyzed / total, 4) if total else 0.0,
            "current_frame": self.current_index,
            "current_time_s": self.current_time_s,
            "current_timecode": (
                format_timecode(self.current_time_s) if self.current_time_s is not None else None
            ),
            "elapsed_s": round(elapsed, 2),
            "rate_per_s": round(rate, 3),
            "eta_s": round(remaining / rate, 1) if rate > 0 else None,
            "step_mode": self.step_mode,
            "step_seconds": self.step_seconds,
            "step_frames": self.step_frames,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "max_width": self.max_width,
            "carry_context": self.carry_context,
            "save_frames": self.save_frames,
            "error": self.error,
            "video_info": self.meta,
        }

    def _emit(self, event: dict[str, Any]) -> None:
        """Hand one progress event to the server's broadcaster. Never raises."""
        if not self.on_event:
            return
        try:
            self.on_event(event)
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"Job event callback failed: {e}")

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    async def _in_capture_thread(self, fn, *args):
        """Run one blocking VideoCapture call on the job's dedicated thread."""
        return await asyncio.get_running_loop().run_in_executor(self._capture_pool, fn, *args)

    async def _run(self) -> None:
        """Drive the whole job: open, sample, analyse, report, close."""
        self.state = STATE_RUNNING
        self.started_at = time.time()
        self._emit({"type": "job_started", **self.status()})
        step_desc = (
            f"every {self.step_seconds}s"
            if self.step_mode == STEP_SECONDS
            else f"every {self.step_frames} frames"
        )
        logger.info(
            f"▶️  Job {self.job_id}: {self.meta['filename']} - "
            f"{len(self.indices)} frames, {step_desc}"
        )

        transcript = self.vlm_service.transcript
        if transcript:
            await transcript.write_event(
                "video_job_start",
                job_id=self.job_id,
                video=self.meta.get("filename"),
                video_duration_s=self.meta.get("duration_s"),
                video_fps=self.meta.get("fps"),
                step_mode=self.step_mode,
                step_seconds=self.step_seconds if self.step_mode == STEP_SECONDS else None,
                step_frames=self.step_frames if self.step_mode == STEP_FRAMES else None,
                planned_frames=len(self.indices),
            )

        cap = None
        try:
            cap = await self._in_capture_thread(self._open_capture)
            position = -1  # Frame index the capture is parked on

            for step_no, frame_index in enumerate(self.indices, start=1):
                if self._cancelled:
                    break

                # Park here while paused, but stay cancellable
                if not self._resume.is_set():
                    await self._resume.wait()
                    if self._cancelled:
                        break

                frame, position = await self._in_capture_thread(
                    self._read_frame_at, cap, frame_index, position
                )
                if frame is None:
                    # A short read means the container ended earlier than its
                    # header claimed - that is the end of the video, not an error
                    logger.info(
                        f"Job {self.job_id}: no frame at index {frame_index}; stopping early"
                    )
                    break

                await self._analyze_frame(step_no, frame_index, frame)

            if self._cancelled:
                self.state = STATE_CANCELLED
            else:
                self.state = STATE_DONE

        except asyncio.CancelledError:
            self.state = STATE_CANCELLED
            raise
        except Exception as e:
            self.state = STATE_FAILED
            self.error = str(e)
            logger.error(f"Job {self.job_id} failed: {e}", exc_info=True)
        finally:
            if cap is not None:
                # Queue the release behind any decode still running on the
                # capture thread; never await it, because this finally may be
                # unwinding a cancellation and must not block on OpenCV
                self._capture_pool.submit(cap.release)
            self._capture_pool.shutdown(wait=False)
            self.finished_at = time.time()
            if transcript:
                await transcript.write_event(
                    "video_job_end",
                    job_id=self.job_id,
                    video=self.meta.get("filename"),
                    state=self.state,
                    analyzed=self.analyzed,
                    failed=self.failed,
                    planned_frames=len(self.indices),
                    elapsed_s=round(self.finished_at - (self.started_at or self.finished_at), 2),
                    error=self.error,
                )
            logger.info(
                f"⏹️  Job {self.job_id} {self.state}: {self.analyzed}/{len(self.indices)} frames "
                f"analysed in {self.finished_at - (self.started_at or self.finished_at):.1f}s"
            )
            self._emit({"type": "job_finished", **self.status()})

    def _open_capture(self) -> cv2.VideoCapture:
        """Open the file (blocking - always called in a worker thread)."""
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.video_path.name}")
        return cap

    def _read_frame_at(self, cap, target: int, position: int):
        """
        Decode the frame at `target`, given the capture sits just past `position`.

        Skipping forward with grab() decodes nothing but keeps the demuxer in
        step, which is both accurate and fast for small strides. For large
        strides a keyframe seek is far cheaper, so we switch once the gap makes
        the seek worth its inaccuracy.

        Returns (frame_or_None, new_position).
        """
        gap = target - position

        if gap > SEEK_THRESHOLD_FRAMES or gap < 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ok, frame = cap.read()
            return (frame if ok else None), target

        # Walk forward, decoding only the frame we actually want
        for _ in range(max(0, gap - 1)):
            if not cap.grab():
                return None, target
        ok, frame = cap.read()
        return (frame if ok else None), target

    async def _analyze_frame(self, step_no: int, frame_index: int, frame) -> None:
        """Describe one sampled frame and report it to the transcript and UI."""
        fps = float(self.meta.get("fps") or FALLBACK_FPS)
        video_time = frame_index / fps
        timecode = format_timecode(video_time)

        self.current_index = frame_index
        self.current_time_s = round(video_time, 3)

        # Resize and encode off the event loop - both are pure CPU work
        prepared = await asyncio.to_thread(_resize_for_vlm, frame, self.max_width)
        image = Image.fromarray(cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB))

        frame_file = None
        if self.save_frames and self.frames_dir:
            frame_file = await asyncio.to_thread(self._save_frame, prepared, step_no, frame_index)

        prompt = self._build_prompt(timecode)
        context = {
            "source": f"video:{self.meta.get('filename')}",
            "job_id": self.job_id,
            "video": self.meta.get("filename"),
            "video_path": str(self.video_path),
            "step": step_no,
            "step_total": len(self.indices),
            "frame_index": frame_index,
            "video_time_s": round(video_time, 3),
            "timecode": timecode,
            "step_mode": self.step_mode,
            "step_seconds": self.step_seconds if self.step_mode == STEP_SECONDS else None,
            "step_frames": self.step_frames if self.step_mode == STEP_FRAMES else None,
        }
        if frame_file:
            context["frame_file"] = frame_file

        response = await self.vlm_service.analyze_frame(image, prompt, context)

        ok = not response.startswith("Error:")
        if ok:
            self.analyzed += 1
            self.last_response = response
        else:
            self.failed += 1

        result = {
            "step": step_no,
            "frame_index": frame_index,
            "video_time_s": round(video_time, 3),
            "timecode": timecode,
            "text": response,
            "ok": ok,
            "latency_ms": round(self.vlm_service.last_inference_time * 1000, 1),
            "frame_url": (f"/api/video/frame/{self.job_id}/{frame_file}" if frame_file else None),
        }
        self.results.append(result)

        self._emit(
            {
                "type": "video_analysis",
                "result": result,
                "job": self.status(),
                "metrics": self.vlm_service.get_metrics(),
            }
        )

    def _save_frame(self, frame, step_no: int, frame_index: int) -> Optional[str]:
        """Write the analysed frame as a JPEG so the UI can show it. Never raises."""
        name = f"{step_no:05d}_f{frame_index}.jpg"
        try:
            ok = cv2.imwrite(
                str(self.frames_dir / name),
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
            )
            return name if ok else None
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"Could not save frame {name}: {e}")
            return None

    def _build_prompt(self, timecode: str) -> Optional[str]:
        """
        Build the prompt for this frame.

        With carry_context on, the previous description is quoted so the model
        can describe the video as a continuing scene rather than as a set of
        unrelated stills. The exact text used is what lands in the transcript.
        """
        base = self.prompt or self.vlm_service.prompt
        if not self.carry_context or not self.last_response:
            return base
        return (
            f"{base}\n\n"
            f"For continuity, the previous sampled moment of this video was described as: "
            f'"{self.last_response}"\n'
            f"Describe the current frame (at {timecode}) on its own terms, "
            f"noting what has changed only when it matters."
        )
