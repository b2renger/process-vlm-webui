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
Process VLM WebUI Server
Serves the web interface, accepts an uploaded video file, and runs the
frame-by-frame VLM analysis job that describes it.

Unlike the live sibling project there is no WebRTC here: the browser plays the
uploaded file back from the server over plain HTTP, while the server decodes the
same file independently and analyses it as fast as the model allows. Progress,
each description and the GPU stats are pushed to the UI over one WebSocket.
"""

import asyncio
import json
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

import aiohttp
from aiohttp import web

from .vlm_service import VLMService
from .gpu_monitor import create_monitor
from .video_analyzer import (
    STEP_FRAMES,
    STEP_SECONDS,
    VIDEO_SUFFIXES,
    VideoAnalysisJob,
    probe_video,
)
from .transcript import (
    DEFAULT_BACKUP_EVERY,
    DEFAULT_BACKUP_KEEP,
    FORMATS,
    FORMAT_JSONL,
    TranscriptWriter,
    claim_transcript_path,
    default_transcript_dir,
    default_transcript_name,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Read uploads this many bytes at a time
UPLOAD_CHUNK = 1024 * 1024
# Default ceiling for an uploaded file (MB); 0 means unlimited
DEFAULT_MAX_UPLOAD_MB = 4096
# Only these characters may appear in an id that reaches the filesystem
SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")

# Global objects
vlm_service = None
websockets = set()  # Track active WebSocket connections
gpu_monitor = None  # GPU monitoring instance
gpu_monitor_task = None  # Background task for GPU monitoring
transcript_writer = None  # TranscriptWriter persisting every VLM analysis to disk
videos = {}  # Registered videos {video_id: metadata dict}
jobs = {}  # All analysis jobs this session {job_id: VideoAnalysisJob}
current_job = None  # The job currently running (only one at a time)
shutting_down = False  # Set on shutdown so new jobs are refused

# Storage locations and defaults, filled in by main()/configure_storage()
videos_dir = None
frames_dir = None
save_frames = True
max_upload_bytes = DEFAULT_MAX_UPLOAD_MB * 1024 * 1024
allow_server_paths = False
default_job_options = {
    "step_mode": STEP_SECONDS,
    "step_seconds": 2.0,
    "step_frames": 30,
    "max_width": 1280,
    "carry_context": False,
}


def is_port_available(port, host="0.0.0.0"):
    """
    Check if a port is available for binding.

    SO_REUSEADDR mirrors what web.run_app() does when it binds for real.
    Without it this probe rejects ports that merely have sockets lingering in
    TIME_WAIT - which is the normal state right after a restart - and a server
    that just stopped could not be started again.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def find_process_using_port(port):
    """
    Find which process is LISTENING on a port (Linux/Unix only).

    -sTCP:LISTEN matters: without it lsof also lists processes that merely have
    a connection *to* the port, so an open browser tab outsranks the real server
    and the "port in use" message tells the user to kill their own browser.
    """
    try:
        # Try lsof first (more reliable)
        result = subprocess.run(
            ["lsof", "-i", f":{port}", "-sTCP:LISTEN", "-t"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0 and result.stdout.strip():
            pid = result.stdout.strip().split()[0]
            # Get process name
            name_result = subprocess.run(
                ["ps", "-p", pid, "-o", "comm="], capture_output=True, text=True, timeout=2
            )
            if name_result.returncode == 0:
                return f"PID {pid} ({name_result.stdout.strip()})"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # lsof not available, try netstat
        try:
            result = subprocess.run(
                ["netstat", "-tulpn"], capture_output=True, text=True, timeout=2
            )
            for line in result.stdout.split("\n"):
                if f":{port}" in line and "LISTEN" in line:
                    parts = line.split()
                    if len(parts) >= 7:
                        return parts[-1]  # PID/Program name
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    return "unknown process"


def find_available_port(start_port=8080, max_attempts=10):
    """Find next available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        if is_port_available(port):
            return port
    return None


async def detect_local_service_and_model():
    """
    Auto-detect available local VLM services and select a model
    Returns: (api_base, model_name) or (None, None) if no service found
    """
    services = [
        ("http://localhost:11434/v1", "Ollama"),
        ("http://localhost:8000/v1", "vLLM"),
        ("http://localhost:30000/v1", "SGLang"),
    ]

    for api_base, service_name in services:
        try:
            # Try to connect to the service
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2)) as session:
                async with session.get(f"{api_base}/models") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        models = data.get("data", [])
                        if models:
                            # Prefer vision models
                            vision_keywords = ["vision", "llava", "llama-3.2", "gemini"]
                            for model in models:
                                model_id = model.get("id", "")
                                if any(keyword in model_id.lower() for keyword in vision_keywords):
                                    logger.info(f"✅ Auto-detected {service_name} at {api_base}")
                                    logger.info(f"   Selected model: {model_id}")
                                    return (api_base, model_id)

                            # If no vision model found, use the first one
                            model_id = models[0].get("id", "")
                            logger.info(f"✅ Auto-detected {service_name} at {api_base}")
                            logger.info(
                                f"   Selected model: {model_id} (vision model preferred but not found)"
                            )
                            return (api_base, model_id)
        except Exception as e:
            logger.debug(f"Service {service_name} not available at {api_base}: {e}")
            continue

    return (None, None)


async def index(request):
    """Serve the main HTML page"""
    content = open(os.path.join(os.path.dirname(__file__), "static", "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def models(request):
    """Return available models from the VLM API"""
    try:
        # Check if custom API base and key are provided in query params
        api_base = request.rel_url.query.get("api_base")
        api_key = request.rel_url.query.get("api_key")

        if api_base:
            # The UI always sends the endpoint it has on screen. When that is
            # the one already in use, query through the live client and flag the
            # running model: otherwise the page finds no current model,
            # auto-selects whichever the backend lists first, and silently
            # replaces the model --model asked for.
            same_endpoint = vlm_service is not None and vlm_service.api_base.rstrip(
                "/"
            ) == api_base.rstrip("/")

            if same_endpoint:
                client = vlm_service.client
                active = vlm_service.model
            else:
                from openai import AsyncOpenAI

                client = AsyncOpenAI(base_url=api_base, api_key=api_key if api_key else "EMPTY")
                active = None

            models_response = await client.models.list()
            models_list = [
                {"id": model.id, "name": model.id, "current": model.id == active}
                for model in models_response.data
            ]
            return web.Response(
                content_type="application/json", text=json.dumps({"models": models_list})
            )
        elif vlm_service:
            # Use the server's VLM service
            models_response = await vlm_service.client.models.list()
            models_list = [
                {"id": model.id, "name": model.id, "current": model.id == vlm_service.model}
                for model in models_response.data
            ]
            return web.Response(
                content_type="application/json", text=json.dumps({"models": models_list})
            )
        else:
            return web.Response(
                content_type="application/json",
                text=json.dumps({"models": [], "error": "VLM service not initialized"}),
            )
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        # Return current model as fallback
        if vlm_service:
            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {
                        "models": [
                            {"id": vlm_service.model, "name": vlm_service.model, "current": True}
                        ]
                    }
                ),
            )
        return web.Response(
            content_type="application/json", text=json.dumps({"models": [], "error": str(e)})
        )


async def detect_services(request):
    """Detect available local VLM services"""
    services = [
        {"name": "Ollama", "url": "http://localhost:11434/v1", "port": 11434, "path": "/api/tags"},
        {"name": "vLLM", "url": "http://localhost:8000/v1", "port": 8000, "path": "/v1/models"},
        {"name": "SGLang", "url": "http://localhost:30000/v1", "port": 30000, "path": "/v1/models"},
    ]

    detected = []

    async def check_service(service):
        """Check if a service is running by probing its endpoint"""
        try:
            timeout = aiohttp.ClientTimeout(total=1.0)  # 1 second timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                url = f"http://localhost:{service['port']}{service['path']}"
                async with session.get(url) as response:
                    if response.status in [200, 404]:  # 404 is ok, means server is running
                        logger.info(f"Detected {service['name']} at {service['url']}")
                        return service
        except (aiohttp.ClientError, asyncio.TimeoutError):
            pass
        return None

    # Check all services concurrently
    results = await asyncio.gather(*[check_service(s) for s in services])
    detected = [s for s in results if s is not None]

    # Default to NVIDIA API Catalog if no local services found
    if not detected:
        detected.append(
            {
                "name": "NVIDIA API Catalog",
                "url": "https://integrate.api.nvidia.com/v1",
                "port": None,
                "path": None,
                "requires_key": True,
            }
        )

    return web.Response(
        content_type="application/json",
        text=json.dumps({"detected": detected, "default": detected[0] if detected else None}),
    )


async def websocket_handler(request):
    """Handle WebSocket connections for analysis updates and configuration"""
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    websockets.add(ws)
    logger.info(f"WebSocket client connected. Total clients: {len(websockets)}")

    try:
        # Send initial message with current server configuration
        await ws.send_json({"type": "status", "text": "Connected to server", "status": "Ready"})

        if vlm_service:
            config = {
                "type": "server_config",
                "model": vlm_service.model,
                "api_base": vlm_service.api_base,
                "prompt": vlm_service.prompt,
                "defaults": default_job_options,
            }
            if transcript_writer:
                config["transcript"] = transcript_writer.stats()
            if current_job:
                config["job"] = current_job.status()
            await ws.send_json(config)

        # Keep connection alive and handle incoming messages
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)

                    if data.get("type") == "update_prompt":
                        new_prompt = data.get("prompt", "").strip()
                        max_tokens = data.get("max_tokens")
                        if new_prompt and vlm_service:
                            vlm_service.update_prompt(new_prompt, max_tokens)
                            logger.info(f"Prompt updated: {new_prompt}, max_tokens: {max_tokens}")

                            # Confirm to client
                            await ws.send_json(
                                {
                                    "type": "prompt_updated",
                                    "prompt": new_prompt,
                                    "max_tokens": max_tokens,
                                }
                            )

                    elif data.get("type") == "update_model":
                        new_model = data.get("model", "").strip()
                        api_base = data.get("api_base", "").strip()
                        api_key = data.get("api_key", "").strip()

                        if new_model and vlm_service:
                            # Update model
                            vlm_service.model = new_model

                            # Update API settings if provided
                            # User may have switched to different service (Ollama <-> vLLM)
                            if api_base:
                                vlm_service.update_api_settings(
                                    api_base, api_key if api_key else None
                                )
                                logger.info(f"Model updated: {new_model}, API: {api_base}")
                            else:
                                logger.info(f"Model updated: {new_model}")

                            # Confirm to client
                            await ws.send_json(
                                {
                                    "type": "model_updated",
                                    "model": new_model,
                                    "api_base": vlm_service.api_base,
                                }
                            )
                except json.JSONDecodeError:
                    logger.error("Invalid JSON from client")
                except Exception as e:
                    logger.error(f"Error handling client message: {e}")
            elif msg.type == web.WSMsgType.ERROR:
                logger.error(f"WebSocket error: {ws.exception()}")
    finally:
        websockets.discard(ws)
        logger.info(f"WebSocket client disconnected. Total clients: {len(websockets)}")

    return ws


def broadcast(payload: dict):
    """
    Push one JSON message to every connected WebSocket client.

    Called from the analysis job as well as the GPU loop, so it must never
    raise: a client that has gone away is simply dropped from the set.
    """
    if not websockets:
        return

    message = json.dumps(payload, default=str)

    dead_websockets = set()
    for ws in websockets:
        try:
            asyncio.create_task(ws.send_str(message))
        except Exception as e:
            logger.error(f"Error sending to websocket: {e}")
            dead_websockets.add(ws)

    websockets.difference_update(dead_websockets)


def broadcast_gpu_stats(stats: dict):
    """Broadcast GPU stats to all connected WebSocket clients"""
    broadcast({"type": "gpu_stats", "stats": stats})


async def gpu_monitor_loop():
    """Background task to periodically collect and broadcast GPU stats"""
    global gpu_monitor

    if not gpu_monitor:
        logger.warning("GPU monitor not initialized, skipping monitoring")
        return

    logger.info("GPU monitoring loop started")

    try:
        while True:
            # Get current stats
            stats = gpu_monitor.get_stats()

            # Update history with current stats
            gpu_monitor.update_history(stats)

            # Add history to stats
            stats["history"] = gpu_monitor.get_history()

            # Broadcast to all connected clients
            broadcast_gpu_stats(stats)

            # Update every 0.25 seconds for detailed GPU monitoring
            await asyncio.sleep(0.25)
    except asyncio.CancelledError:
        logger.info("GPU monitoring loop cancelled")
    except Exception as e:
        logger.error(f"Error in GPU monitoring loop: {e}")


def _json_response(payload: dict, status: int = 200) -> web.Response:
    """Small helper for the JSON API responses"""
    return web.Response(
        status=status, content_type="application/json", text=json.dumps(payload, default=str)
    )


def configure_storage(
    videos_path,
    frames_path,
    keep_frames: bool = True,
    max_upload_mb: int = DEFAULT_MAX_UPLOAD_MB,
    server_paths: bool = False,
):
    """
    Decide where uploads and extracted frames live, and create the directories.

    Called once from main() before the app starts. Failing here is fatal on
    purpose - an app that cannot store the video it was asked to analyse has
    nothing useful to do.
    """
    global videos_dir, frames_dir, save_frames, max_upload_bytes, allow_server_paths

    videos_dir = Path(videos_path).expanduser().resolve()
    frames_dir = Path(frames_path).expanduser().resolve()
    videos_dir.mkdir(parents=True, exist_ok=True)
    if keep_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    save_frames = bool(keep_frames)
    max_upload_bytes = max(0, int(max_upload_mb)) * 1024 * 1024
    allow_server_paths = bool(server_paths)

    logger.info(f"🎞️  Videos:  {videos_dir}")
    logger.info(f"🖼️  Frames:  {frames_dir if save_frames else 'disabled (--no-save-frames)'}")


def register_video(path, video_id=None, uploaded: bool = False) -> dict:
    """
    Probe a video file and remember it under an id the API can refer to.

    Raises ValueError when the file is not decodable, so both the upload path
    and the --video startup path reject a bad file the same way.
    """
    path = Path(path).expanduser().resolve()
    meta = probe_video(path)
    meta["video_id"] = video_id or f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    meta["uploaded"] = uploaded
    meta["url"] = f"/api/video/file/{meta['video_id']}"
    meta["registered_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    videos[meta["video_id"]] = meta
    logger.info(
        f"🎬 Registered video {meta['video_id']}: {meta['filename']} "
        f"({meta['width']}x{meta['height']}, {meta['duration_timecode']}, {meta['fps']}fps)"
    )
    return meta


def _safe_id(value: str) -> bool:
    """Reject ids that could escape the storage directory."""
    return bool(value) and bool(SAFE_ID.match(value))


async def video_upload(request):
    """
    Accept a video file upload.

    POST /api/video/upload   (multipart/form-data, field name "file")

    The body is streamed straight to disk in chunks so a multi-gigabyte file
    never has to fit in memory, and the size ceiling is enforced as it arrives
    rather than after the fact.
    """
    if shutting_down:
        return _json_response({"error": "Server is shutting down"}, status=503)

    try:
        reader = await request.multipart()
    except Exception as e:
        return _json_response({"error": f"Not a multipart upload: {e}"}, status=400)

    field = await reader.next()
    while field is not None and field.name != "file":
        field = await reader.next()

    if field is None:
        return _json_response({"error": "No 'file' part in the upload"}, status=400)

    original = Path(field.filename or "video.mp4").name
    suffix = Path(original).suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        return _json_response(
            {
                "error": f"Unsupported file type '{suffix or original}'. "
                f"Expected one of: {', '.join(VIDEO_SUFFIXES)}"
            },
            status=400,
        )

    video_id = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    dest = videos_dir / f"{video_id}{suffix}"

    size = 0
    try:
        with open(dest, "wb") as f:
            while True:
                chunk = await field.read_chunk(UPLOAD_CHUNK)
                if not chunk:
                    break
                size += len(chunk)
                if max_upload_bytes and size > max_upload_bytes:
                    raise ValueError(
                        f"File exceeds the {max_upload_bytes // (1024 * 1024)} MB upload limit"
                    )
                await asyncio.to_thread(f.write, chunk)
    except ValueError as e:
        dest.unlink(missing_ok=True)
        logger.warning(f"Upload rejected: {e}")
        return _json_response({"error": str(e)}, status=413)
    except Exception as e:
        dest.unlink(missing_ok=True)
        logger.error(f"Upload failed: {e}", exc_info=True)
        return _json_response({"error": f"Upload failed: {e}"}, status=500)

    if size == 0:
        dest.unlink(missing_ok=True)
        return _json_response({"error": "Uploaded file is empty"}, status=400)

    # Probe before accepting - a file OpenCV cannot open is rejected now, with
    # a clear message, rather than failing later inside a job
    try:
        meta = await asyncio.to_thread(register_video, dest, video_id, True)
    except ValueError as e:
        dest.unlink(missing_ok=True)
        return _json_response({"error": str(e)}, status=400)

    meta = dict(meta)
    meta["original_filename"] = original
    videos[video_id]["original_filename"] = original
    logger.info(f"⬆️  Uploaded {original} as {video_id} ({size / 1024 / 1024:.1f} MB)")

    return _json_response({"status": "ok", "video": meta})


async def video_register(request):
    """
    Register a video that already sits on the server's filesystem.

    POST /api/video/register
    Body: {"path": "/data/clips/site-survey.mp4"}

    Disabled unless the server was started with --allow-server-paths, because
    it otherwise lets any client on the network read any file this process can.
    """
    if not allow_server_paths:
        return _json_response(
            {"error": "Server-side paths are disabled. Restart with --allow-server-paths."},
            status=403,
        )

    try:
        data = await request.json()
    except json.JSONDecodeError:
        return _json_response({"error": "Invalid JSON body"}, status=400)

    raw = (data.get("path") or "").strip()
    if not raw:
        return _json_response({"error": "Missing 'path'"}, status=400)

    path = Path(raw).expanduser()
    if not path.is_file():
        return _json_response({"error": f"No such file: {raw}"}, status=404)

    try:
        meta = await asyncio.to_thread(register_video, path, None, False)
    except ValueError as e:
        return _json_response({"error": str(e)}, status=400)

    return _json_response({"status": "ok", "video": meta})


async def video_list(request):
    """
    List every video registered this session.

    GET /api/video/list
    """
    return _json_response(
        {
            "videos": sorted(videos.values(), key=lambda v: v["registered_at"], reverse=True),
            "count": len(videos),
            "allow_server_paths": allow_server_paths,
        }
    )


async def video_file(request):
    """
    Serve a registered video back for playback in the browser.

    GET /api/video/file/{video_id}
    """
    video_id = request.match_info.get("video_id", "")
    meta = videos.get(video_id)
    if not meta:
        return _json_response({"error": "Unknown video id"}, status=404)

    path = Path(meta["path"])
    if not path.exists():
        return _json_response({"error": "Video file is no longer on disk"}, status=410)

    # Range support matters here: without it the browser cannot seek the video
    return web.FileResponse(path, headers={"Accept-Ranges": "bytes"})


async def video_frame(request):
    """
    Serve one analysed frame image.

    GET /api/video/frame/{job_id}/{name}
    """
    job_id = request.match_info.get("job_id", "")
    name = request.match_info.get("name", "")

    # Both parts land in a filesystem path, so neither may contain traversal
    if not _safe_id(job_id) or not _safe_id(name):
        return _json_response({"error": "Invalid frame reference"}, status=400)

    path = frames_dir / job_id / name
    if not path.is_file():
        return _json_response({"error": "Frame not found"}, status=404)

    return web.FileResponse(path)


def _job_option(data: dict, key: str, cast, default):
    """Read one job option from the request body, falling back to the default."""
    if key not in data or data[key] is None or data[key] == "":
        return default
    return cast(data[key])


async def video_start(request):
    """
    Start analysing a registered video.

    POST /api/video/start
    Body: {
      "video_id": "...",           # required
      "step_mode": "seconds",      # or "frames"
      "step_seconds": 2.0,         # time step between analysed frames
      "step_frames": 30,           # frame step, when step_mode is "frames"
      "start_s": 0, "end_s": null, # optional range within the video
      "max_frames": 0,             # optional cap, 0 = no cap
      "max_width": 1280,           # downscale before inference
      "carry_context": false,      # feed the previous description back in
      "prompt": "...",             # optional override for this job
      "max_tokens": 512
    }
    """
    global current_job

    if shutting_down:
        return _json_response({"error": "Server is shutting down"}, status=503)

    if vlm_service is None:
        return _json_response({"error": "VLM service not initialized"}, status=503)

    if current_job is not None and current_job.status()["active"]:
        return _json_response(
            {
                "error": "A job is already running. Stop it first.",
                "job": current_job.status(),
            },
            status=409,
        )

    try:
        data = await request.json()
    except json.JSONDecodeError:
        return _json_response({"error": "Invalid JSON body"}, status=400)

    meta = videos.get((data.get("video_id") or "").strip())
    if not meta:
        return _json_response({"error": "Unknown or missing video_id"}, status=404)

    if not Path(meta["path"]).exists():
        return _json_response({"error": "Video file is no longer on disk"}, status=410)

    # A per-job prompt is applied to the shared service so the UI, the model and
    # the transcript all agree on what was asked
    prompt = (data.get("prompt") or "").strip()
    if prompt:
        max_tokens = data.get("max_tokens")
        vlm_service.update_prompt(prompt, int(max_tokens) if max_tokens else None)

    try:
        job = VideoAnalysisJob(
            meta["path"],
            vlm_service,
            video_id=meta["video_id"],
            display_name=meta.get("original_filename") or meta["filename"],
            step_mode=_job_option(data, "step_mode", str, default_job_options["step_mode"]),
            step_seconds=_job_option(
                data, "step_seconds", float, default_job_options["step_seconds"]
            ),
            step_frames=_job_option(data, "step_frames", int, default_job_options["step_frames"]),
            start_s=_job_option(data, "start_s", float, 0.0),
            end_s=_job_option(data, "end_s", float, None),
            max_frames=_job_option(data, "max_frames", int, 0),
            max_width=_job_option(data, "max_width", int, default_job_options["max_width"]),
            carry_context=bool(
                _job_option(data, "carry_context", bool, default_job_options["carry_context"])
            ),
            prompt=prompt or None,
            save_frames=save_frames,
            frames_dir=frames_dir,
            on_event=broadcast,
        )
    except ValueError as e:
        return _json_response({"error": str(e)}, status=400)

    if not job.indices:
        return _json_response(
            {"error": "The chosen step and range select no frames from this video"}, status=400
        )

    jobs[job.job_id] = job
    current_job = job
    job.start()

    return _json_response({"status": "started", "job": job.status()})


async def video_stop(request):
    """
    Cancel the running job.

    POST /api/video/stop
    """
    if current_job is None:
        return _json_response({"error": "No job to stop"}, status=404)

    await current_job.cancel()
    status = current_job.status()
    logger.info(f"Job {current_job.job_id} cancelled by client")
    return _json_response({"status": "stopped", "job": status})


async def video_pause(request):
    """
    Pause or resume the running job.

    POST /api/video/pause
    Body: {"paused": true}
    """
    if current_job is None:
        return _json_response({"error": "No job is running"}, status=404)

    try:
        data = await request.json()
    except json.JSONDecodeError:
        data = {}

    if data.get("paused", True):
        current_job.pause()
    else:
        current_job.resume()

    return _json_response({"status": "ok", "job": current_job.status()})


async def video_status(request):
    """
    Report the state of a job (the running one by default).

    GET /api/video/status[?job_id=...]
    """
    job_id = request.rel_url.query.get("job_id")
    job = jobs.get(job_id) if job_id else current_job

    if job is None:
        return _json_response({"active": False, "job": None, "jobs": len(jobs)})

    return _json_response({"active": job.status()["active"], "job": job.status()})


async def video_results(request):
    """
    Return the descriptions produced so far by a job.

    GET /api/video/results?job_id=...&offset=0&limit=500

    The UI uses this to rebuild the timeline after a reload; live updates
    arrive over the WebSocket instead.
    """
    job_id = request.rel_url.query.get("job_id")
    job = jobs.get(job_id) if job_id else current_job

    if job is None:
        return _json_response({"results": [], "count": 0, "job": None})

    try:
        offset = max(0, int(request.rel_url.query.get("offset", 0)))
    except ValueError:
        offset = 0
    try:
        limit = max(1, min(2000, int(request.rel_url.query.get("limit", 500))))
    except ValueError:
        limit = 500

    window = job.results[offset : offset + limit]
    return _json_response(
        {
            "results": window,
            "count": len(window),
            "total": len(job.results),
            "offset": offset,
            "job": job.status(),
        }
    )


async def video_jobs(request):
    """
    List every job created this session, newest first.

    GET /api/video/jobs
    """
    return _json_response(
        {
            "jobs": [job.status() for job in reversed(list(jobs.values()))],
            "count": len(jobs),
            "current": current_job.job_id if current_job else None,
        }
    )


async def video_delete(request):
    """
    Forget a video and delete its uploaded file.

    POST /api/video/delete
    Body: {"video_id": "..."}

    Only files this server stored are deleted; a video registered from a
    server-side path is forgotten but left alone on disk.
    """
    try:
        data = await request.json()
    except json.JSONDecodeError:
        return _json_response({"error": "Invalid JSON body"}, status=400)

    video_id = (data.get("video_id") or "").strip()
    meta = videos.get(video_id)
    if not meta:
        return _json_response({"error": "Unknown video id"}, status=404)

    if current_job and current_job.video_id == video_id and current_job.status()["active"]:
        return _json_response({"error": "That video is being analysed right now"}, status=409)

    removed = False
    if meta.get("uploaded"):
        try:
            await asyncio.to_thread(Path(meta["path"]).unlink, True)
            removed = True
        except OSError as e:
            logger.warning(f"Could not delete {meta['path']}: {e}")

    videos.pop(video_id, None)
    return _json_response({"status": "ok", "video_id": video_id, "file_removed": removed})


def _tail_lines(path, limit: int, max_bytes: int = 262144) -> list:
    """
    Read up to `limit` trailing lines without loading the whole file.

    Only the last `max_bytes` of the file are scanned, so this stays cheap on
    long-running sessions. A partial first line from the scan window is
    discarded rather than returned malformed.
    """
    try:
        size = os.path.getsize(path)
    except OSError:
        return []

    start = max(0, size - max_bytes)
    try:
        with open(path, "rb") as f:
            f.seek(start)
            chunk = f.read()
    except OSError:
        return []

    text = chunk.decode("utf-8", errors="replace")
    lines = text.splitlines()
    # Drop a line truncated by the seek offset
    if start > 0 and lines:
        lines = lines[1:]
    return [ln for ln in lines if ln.strip()][-limit:]


async def transcript_status(request):
    """
    Return transcript writer state.

    GET /api/transcript/status
    """
    if not transcript_writer:
        return _json_response({"enabled": False, "ready": False, "available": False})

    stats = transcript_writer.stats()
    stats["available"] = True
    return _json_response(stats)


async def transcript_recent(request):
    """
    Return the most recent transcript records (newest last).

    GET /api/transcript/recent?limit=50
    """
    if not transcript_writer:
        return _json_response({"records": [], "available": False})

    try:
        limit = max(1, min(500, int(request.rel_url.query.get("limit", 50))))
    except ValueError:
        limit = 50

    lines = await asyncio.to_thread(_tail_lines, transcript_writer.path, limit)

    if transcript_writer.format == FORMAT_JSONL:
        records = []
        for line in lines:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # A torn trailing line is expected after a hard kill - skip it
                continue
    else:
        records = [{"text": line} for line in lines]

    return _json_response({"records": records, "available": True, "count": len(records)})


async def transcript_download(request):
    """
    Download the current transcript file.

    GET /api/transcript/download
    """
    if not transcript_writer or not transcript_writer.path.exists():
        return _json_response({"error": "No transcript file available"}, status=404)

    return web.FileResponse(
        transcript_writer.path,
        headers={
            "Content-Disposition": f'attachment; filename="{transcript_writer.path.name}"',
            "Content-Type": (
                "application/x-ndjson" if transcript_writer.format == FORMAT_JSONL else "text/plain"
            ),
        },
    )


async def transcript_backup(request):
    """
    Force a backup snapshot right now.

    POST /api/transcript/backup
    """
    if not transcript_writer:
        return _json_response({"error": "Transcript not configured"}, status=404)

    path = await transcript_writer.backup_now()
    if not path:
        return _json_response(
            {"status": "skipped", "reason": "nothing to back up", **transcript_writer.stats()}
        )
    return _json_response({"status": "ok", "backup": path, **transcript_writer.stats()})


async def transcript_config(request):
    """
    Update transcript settings at runtime.

    POST /api/transcript/config
    Body: {"enabled": true, "backup_every": 25}
    """
    if not transcript_writer:
        return _json_response({"error": "Transcript not configured"}, status=404)

    try:
        data = await request.json()
    except json.JSONDecodeError:
        return _json_response({"error": "Invalid JSON body"}, status=400)

    if "enabled" in data:
        want = bool(data["enabled"])
        was_enabled = transcript_writer.enabled
        if was_enabled and not want:
            # Mark the pause before stopping, otherwise the marker is dropped
            # by the very pause it documents and the gap is unexplained
            await transcript_writer.write_event("recording_paused")
            transcript_writer.set_enabled(False)
        elif not was_enabled and want:
            transcript_writer.set_enabled(True)
            await transcript_writer.write_event("recording_resumed")

    if "backup_every" in data:
        try:
            every = int(data["backup_every"])
        except (TypeError, ValueError):
            return _json_response({"error": "backup_every must be an integer"}, status=400)
        if not 0 <= every <= 10000:
            return _json_response({"error": "backup_every must be 0-10000"}, status=400)
        transcript_writer.backup_every = every
        logger.info(f"📝 Transcript backup interval set to {every} records")

    return _json_response({"status": "ok", **transcript_writer.stats()})


async def on_startup(app):
    """Initialize resources on server startup"""
    global gpu_monitor, gpu_monitor_task

    # Initialize GPU monitor
    try:
        gpu_monitor = create_monitor()
        logger.info("GPU monitor initialized")
    except Exception as e:
        logger.error(f"Failed to initialize GPU monitor: {e}")
        gpu_monitor = None

    # Start GPU monitoring background task
    if gpu_monitor:
        gpu_monitor_task = asyncio.create_task(gpu_monitor_loop())
        logger.info("GPU monitoring task started")

    # Open the transcript with a session header so each run is identifiable
    if transcript_writer:
        from . import __version__

        await transcript_writer.write_event(
            "session_start",
            version=__version__,
            app="process-vlm-webui",
            model=vlm_service.model if vlm_service else None,
            api_base=vlm_service.api_base if vlm_service else None,
            prompt=vlm_service.prompt if vlm_service else None,
        )


async def on_shutdown(app):
    """
    Stop background work and refuse new jobs.

    The running analysis job is cancelled here rather than in on_cleanup: it
    holds an open VideoCapture and will keep calling the VLM otherwise, and
    unlike a WebRTC negotiation it is safe to interrupt at any frame boundary.
    """
    global gpu_monitor, gpu_monitor_task, shutting_down

    logger.info("Shutting down server...")
    shutting_down = True

    # Stop the analysis job first - it is the only thing still producing
    # transcript records, and the transcript is closed in on_cleanup
    if current_job is not None and current_job.status()["active"]:
        logger.info(f"Cancelling running job {current_job.job_id}")
        await current_job.cancel()

    # Stop GPU monitoring task
    if gpu_monitor_task:
        gpu_monitor_task.cancel()
        try:
            await gpu_monitor_task
        except asyncio.CancelledError:
            pass
        logger.info("GPU monitoring task stopped")

    # Cleanup GPU monitor
    if gpu_monitor:
        gpu_monitor.cleanup()
        logger.info("GPU monitor cleaned up")

    # Close all websockets
    for ws in list(websockets):
        await ws.close()
    websockets.clear()

    logger.info("Shutdown signalled; waiting for in-flight requests")


async def on_cleanup(app):
    """Final teardown, after aiohttp has drained in-flight request handlers."""
    global shutting_down

    # Close the transcript last, once no more analyses can be produced
    if transcript_writer:
        await transcript_writer.close()

    shutting_down = False
    logger.info("Cleanup complete")


async def create_app(test_mode=False):
    """
    Create and configure the aiohttp web application.

    Args:
        test_mode: If True, skip GPU monitoring and use test configuration

    Returns:
        Configured web.Application instance
    """
    # Uploads are streamed chunk by chunk, but aiohttp still refuses a body
    # larger than client_max_size before the handler ever sees it - so this has
    # to be at least as generous as our own upload ceiling.
    body_limit = (max_upload_bytes or DEFAULT_MAX_UPLOAD_MB * 1024 * 1024) + UPLOAD_CHUNK
    app = web.Application(client_max_size=body_limit)

    app.router.add_get("/", index)
    app.router.add_get("/models", models)
    app.router.add_get("/detect-services", detect_services)
    app.router.add_get("/ws", websocket_handler)

    # Video endpoints
    app.router.add_post("/api/video/upload", video_upload)
    app.router.add_post("/api/video/register", video_register)
    app.router.add_get("/api/video/list", video_list)
    app.router.add_get("/api/video/file/{video_id}", video_file)
    app.router.add_get("/api/video/frame/{job_id}/{name}", video_frame)
    app.router.add_post("/api/video/start", video_start)
    app.router.add_post("/api/video/stop", video_stop)
    app.router.add_post("/api/video/pause", video_pause)
    app.router.add_get("/api/video/status", video_status)
    app.router.add_get("/api/video/results", video_results)
    app.router.add_get("/api/video/jobs", video_jobs)
    app.router.add_post("/api/video/delete", video_delete)

    # Transcript endpoints
    app.router.add_get("/api/transcript/status", transcript_status)
    app.router.add_get("/api/transcript/recent", transcript_recent)
    app.router.add_get("/api/transcript/download", transcript_download)
    app.router.add_post("/api/transcript/backup", transcript_backup)
    app.router.add_post("/api/transcript/config", transcript_config)

    # Serve static files (images, etc.)
    # Always serve from static/images within the package (works for both pip and dev installs)
    images_dir = os.path.join(os.path.dirname(__file__), "static", "images")
    images_dir = os.path.abspath(images_dir)

    if os.path.exists(images_dir):
        app.router.add_static("/images", images_dir, name="images")
        logger.info(f"Serving static files from: {images_dir}")
    else:
        logger.warning(f"⚠️  Static images directory not found: {images_dir}")

    # Serve favicon files
    favicon_dir = os.path.join(os.path.dirname(__file__), "static", "favicon")
    favicon_dir = os.path.abspath(favicon_dir)

    if os.path.exists(favicon_dir):
        app.router.add_static("/favicon", favicon_dir, name="favicon")
        logger.info(f"Serving favicon files from: {favicon_dir}")
    else:
        logger.warning(f"⚠️  Favicon directory not found: {favicon_dir}")

    if not test_mode:
        app.on_startup.append(on_startup)
        app.on_shutdown.append(on_shutdown)
        app.on_cleanup.append(on_cleanup)

    return app


def get_app_config_dir():
    """Get the application config directory following OS conventions"""
    import os
    from pathlib import Path

    # Follow XDG Base Directory spec on Linux, use OS-appropriate paths elsewhere
    if os.name == "posix":
        if "darwin" in os.sys.platform.lower():
            # macOS
            config_dir = Path.home() / "Library" / "Application Support" / "process-vlm-webui"
        else:
            # Linux/Unix (including Jetson)
            config_dir = (
                Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
                / "process-vlm-webui"
            )
    else:
        # Windows
        config_dir = Path(os.environ.get("APPDATA", Path.home())) / "process-vlm-webui"

    # Create directory if it doesn't exist
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def build_transcript_writer(args):
    """
    Build the TranscriptWriter from parsed CLI arguments.

    Returns None when recording is disabled or the target is unwritable, in
    which case the server runs exactly as it did before this feature existed.
    """
    if args.no_transcript:
        logger.info("📝 Transcript recording disabled (--no-transcript)")
        return None

    fmt = args.transcript_format
    if args.transcript_file:
        # An explicit path is honoured as-is, and appends if it already exists
        path = Path(args.transcript_file).expanduser()
    else:
        directory = (
            Path(args.transcript_dir).expanduser()
            if args.transcript_dir
            else default_transcript_dir()
        )
        # Claim the name atomically so concurrent instances can never end
        # up appending into the same file
        try:
            path = claim_transcript_path(directory, fmt)
        except OSError:
            # Let prepare() produce the single, consistent error message
            path = directory / default_transcript_name(fmt)

    writer = TranscriptWriter(
        path=path,
        fmt=fmt,
        backup_every=args.transcript_backup_every,
        backup_keep=args.transcript_backup_keep,
        max_bytes=args.transcript_max_bytes,
        fsync=not args.no_transcript_fsync,
    )
    writer.prepare()
    return writer


def generate_self_signed_cert(cert_path="cert.pem", key_path="key.pem"):
    """Generate a self-signed SSL certificate if it doesn't exist"""
    import subprocess
    import os

    if os.path.exists(cert_path) and os.path.exists(key_path):
        return True

    logger.info("🔐 Generating self-signed SSL certificate...")
    logger.info(f"   Saving to: {os.path.dirname(os.path.abspath(cert_path)) or '.'}")
    try:
        subprocess.run(
            [
                "openssl",
                "req",
                "-x509",
                "-newkey",
                "rsa:4096",
                "-nodes",
                "-out",
                cert_path,
                "-keyout",
                key_path,
                "-days",
                "365",
                "-subj",
                "/CN=localhost",
            ],
            check=True,
            capture_output=True,
        )
        logger.info(f"✅ Generated {cert_path} and {key_path}")
        return True
    except FileNotFoundError:
        logger.warning("⚠️  openssl not found - cannot auto-generate certificates")
        logger.warning(
            "⚠️  Install openssl: sudo apt install openssl (Linux) or brew install openssl (Mac)"
        )
        return False
    except subprocess.CalledProcessError as e:
        logger.warning(f"⚠️  Failed to generate certificates: {e}")
        return False


def default_videos_dir() -> Path:
    """Where uploaded videos are stored, alongside the transcripts."""
    return default_transcript_dir().parent / "videos"


def default_frames_dir() -> Path:
    """Where analysed frames are stored, alongside the transcripts."""
    return default_transcript_dir().parent / "frames"


def main():
    """Main entry point"""
    import argparse
    import ssl
    from . import __version__

    parser = argparse.ArgumentParser(
        description="Process VLM WebUI - describe a video file with a Vision Language Model",
        epilog="Examples:\n"
        "  Ollama:  process-vlm-webui --model gemma3:27b --api-base http://localhost:11434/v1\n"
        "  vLLM:    process-vlm-webui --model llama-3.2-11b-vision-instruct "
        "--api-base http://localhost:8000/v1\n"
        "  Step:    process-vlm-webui --step 5          # describe every 5 seconds of video\n"
        "  Preload: process-vlm-webui --video clip.mp4  # offer a local file in the UI\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8091, help="Port to bind to (default: 8091)")
    parser.add_argument(
        "--auto-port",
        action="store_true",
        help="Automatically find available port if default is taken",
    )
    parser.add_argument(
        "--model", help="VLM model name (optional, will auto-detect if not specified)"
    )
    parser.add_argument(
        "--api-base", help="VLM API base URL (optional, will auto-detect or use NVIDIA NGC)"
    )
    parser.add_argument(
        "--api-key",
        default="EMPTY",
        help="API key - use 'EMPTY' for local servers, required for NVIDIA NGC/OpenAI "
        "(default: EMPTY)",
    )
    parser.add_argument(
        "--prompt",
        default="Describe what you see in this image in one sentence.",
        help="Prompt to send to VLM for each sampled frame",
    )

    # Video sampling options
    video_group = parser.add_argument_group("video analysis")
    video_group.add_argument(
        "--step",
        type=float,
        default=2.0,
        metavar="SECONDS",
        help="Seconds of video between analysed frames (default: 2.0)",
    )
    video_group.add_argument(
        "--step-frames",
        type=int,
        default=0,
        metavar="N",
        help="Sample every Nth frame instead of a time step (overrides --step)",
    )
    video_group.add_argument(
        "--max-width",
        type=int,
        default=1280,
        metavar="PX",
        help="Downscale frames to this width before inference, 0 to keep the original size (default: 1280)",
    )
    video_group.add_argument(
        "--carry-context",
        action="store_true",
        help="Include the previous description in each prompt so the model "
        "describes the video as a continuing scene",
    )
    video_group.add_argument(
        "--video",
        default=None,
        metavar="PATH",
        help="Register a local video file at startup so it is ready in the UI",
    )
    video_group.add_argument(
        "--videos-dir",
        default=None,
        help=f"Directory for uploaded videos (default: {default_videos_dir()})",
    )
    video_group.add_argument(
        "--frames-dir",
        default=None,
        help=f"Directory for analysed frame images (default: {default_frames_dir()})",
    )
    video_group.add_argument(
        "--no-save-frames",
        action="store_true",
        help="Do not keep a JPEG of each analysed frame (saves disk, loses the UI thumbnails)",
    )
    video_group.add_argument(
        "--max-upload-mb",
        type=int,
        default=DEFAULT_MAX_UPLOAD_MB,
        metavar="MB",
        help=f"Largest accepted upload, 0 for unlimited (default: {DEFAULT_MAX_UPLOAD_MB})",
    )
    video_group.add_argument(
        "--allow-server-paths",
        action="store_true",
        help="Let clients analyse any file path on this machine (off by default)",
    )

    # Get default SSL cert paths (platform-specific)
    default_config_dir = get_app_config_dir()
    default_cert_path = str(default_config_dir / "cert.pem")
    default_key_path = str(default_config_dir / "key.pem")

    parser.add_argument(
        "--ssl",
        action="store_true",
        help="Serve over HTTPS (certificates are auto-generated if missing). "
        "Plain HTTP is the default - no camera permission is needed here.",
    )
    parser.add_argument(
        "--ssl-cert",
        default=None,
        help=f"Path to SSL certificate file (default: {default_cert_path})",
    )
    parser.add_argument(
        "--ssl-key",
        default=None,
        help=f"Path to SSL private key file (default: {default_key_path})",
    )

    # Transcript (analysis logging) options
    transcript_group = parser.add_argument_group("transcript")
    transcript_group.add_argument(
        "--no-transcript",
        action="store_true",
        help="Disable saving VLM analyses to disk (recording is on by default)",
    )
    transcript_group.add_argument(
        "--transcript-dir",
        default=None,
        help=f"Directory for transcript files (default: {default_transcript_dir()})",
    )
    transcript_group.add_argument(
        "--transcript-file",
        default=None,
        help="Full path of the transcript file (overrides --transcript-dir; "
        "appends if the file already exists)",
    )
    transcript_group.add_argument(
        "--transcript-format",
        default=FORMAT_JSONL,
        choices=list(FORMATS),
        help=f"Transcript format: jsonl (structured) or txt (human readable) "
        f"(default: {FORMAT_JSONL})",
    )
    transcript_group.add_argument(
        "--transcript-backup-every",
        type=int,
        default=DEFAULT_BACKUP_EVERY,
        metavar="N",
        help=f"Write a backup snapshot every N records, 0 to disable "
        f"(default: {DEFAULT_BACKUP_EVERY})",
    )
    transcript_group.add_argument(
        "--transcript-backup-keep",
        type=int,
        default=DEFAULT_BACKUP_KEEP,
        metavar="K",
        help=f"Number of backup snapshots to retain, 0 to keep all "
        f"(default: {DEFAULT_BACKUP_KEEP})",
    )
    transcript_group.add_argument(
        "--transcript-max-bytes",
        type=int,
        default=0,
        metavar="BYTES",
        help="Rotate the transcript once it exceeds this size, 0 to disable (default: 0)",
    )
    transcript_group.add_argument(
        "--no-transcript-fsync",
        action="store_true",
        help="Skip fsync after each record (faster, but a crash can lose recent analyses)",
    )

    args = parser.parse_args()

    # Resolve the listening port first, before claiming any resources, so a
    # doomed start never leaves an orphaned transcript file behind
    if not is_port_available(args.port, args.host):
        holder = find_process_using_port(args.port)
        if args.auto_port:
            new_port = find_available_port(args.port)
            if new_port is None:
                logger.error(f"❌ Port {args.port} is in use by {holder}")
                logger.error(f"❌ No free port found in {args.port}-{args.port + 9}")
                sys.exit(1)
            logger.warning(f"⚠️  Port {args.port} is in use by {holder}")
            logger.info(f"   --auto-port: using port {new_port} instead")
            args.port = new_port
        else:
            logger.error(f"❌ Port {args.port} is already in use by {holder}")
            logger.error("   Pick another with --port <N>, or add --auto-port to")
            logger.error("   find the next free port automatically")
            sys.exit(1)

    # Set default SSL cert paths to config directory if not specified
    if args.ssl_cert is None:
        args.ssl_cert = str(get_app_config_dir() / "cert.pem")
    if args.ssl_key is None:
        args.ssl_key = str(get_app_config_dir() / "key.pem")

    # Storage for uploads and frames
    try:
        configure_storage(
            args.videos_dir or default_videos_dir(),
            args.frames_dir or default_frames_dir(),
            keep_frames=not args.no_save_frames,
            max_upload_mb=args.max_upload_mb,
            server_paths=args.allow_server_paths,
        )
    except OSError as e:
        logger.error(f"❌ Cannot create storage directories: {e}")
        sys.exit(1)

    # Default sampling options, echoed to the UI when it connects
    if args.step_frames and args.step_frames > 0:
        default_job_options["step_mode"] = STEP_FRAMES
        default_job_options["step_frames"] = args.step_frames
    else:
        default_job_options["step_mode"] = STEP_SECONDS
    default_job_options["step_seconds"] = args.step
    default_job_options["max_width"] = args.max_width
    default_job_options["carry_context"] = args.carry_context

    # Auto-detect service and model if not specified
    api_base = args.api_base
    model = args.model
    api_key = args.api_key

    if not model or not api_base:
        logger.info("No model/API specified, auto-detecting local services...")
        detected_api_base, detected_model = asyncio.run(detect_local_service_and_model())

        if detected_api_base and detected_model:
            if not api_base:
                api_base = detected_api_base
            if not model:
                model = detected_model
        else:
            # Fall back to NVIDIA NGC
            logger.warning("⚠️  No local VLM service found (Ollama, vLLM, SGLang)")
            logger.info("📡 Falling back to NVIDIA API Catalog")
            logger.info("   You'll need an API key from: https://build.nvidia.com")
            if not api_base:
                api_base = "https://integrate.api.nvidia.com/v1"
            if not model:
                model = "meta/llama-3.2-11b-vision-instruct"
            if api_key == "EMPTY":
                logger.warning("⚠️  API key required for NVIDIA API Catalog")
                logger.warning("   Set with: --api-key YOUR_API_KEY")
                logger.warning("   Or use WebUI to configure API settings after starting")

    # Initialize the transcript writer before the VLM service so every analysis
    # produced by this process is captured from the very first frame
    global transcript_writer
    transcript_writer = build_transcript_writer(args)

    # Initialize VLM service
    global vlm_service
    vlm_service = VLMService(
        model=model,
        api_base=api_base,
        api_key=api_key,
        prompt=args.prompt,
        transcript=transcript_writer,
    )

    service_name = "Local" if "localhost" in api_base or "127.0.0.1" in api_base else "Cloud"
    logger.info("Initialized VLM service:")
    logger.info(f"  Model: {model}")
    logger.info(f"  API: {api_base} ({service_name})")
    logger.info(f"  Prompt: {args.prompt}")
    step_desc = (
        f"every {args.step_frames} frames"
        if default_job_options["step_mode"] == STEP_FRAMES
        else f"every {args.step}s"
    )
    logger.info(f"  Default sampling: {step_desc}, frames downscaled to {args.max_width}px")

    # Register a startup video if one was given
    if args.video:
        try:
            register_video(Path(args.video), uploaded=False)
        except (ValueError, OSError) as e:
            logger.error(f"❌ --video {args.video}: {e}")
            sys.exit(1)

    # Create web application using create_app
    app = asyncio.run(create_app(test_mode=False))

    # Setup SSL (opt-in; unlike the live app there is no camera to authorize)
    ssl_context = None
    protocol = "http"
    if args.ssl:
        if not os.path.exists(args.ssl_cert) or not os.path.exists(args.ssl_key):
            if not generate_self_signed_cert(args.ssl_cert, args.ssl_key):
                logger.error("❌ Cannot enable HTTPS without certificates")
                logger.error("   Install openssl, or drop --ssl to serve over HTTP")
                sys.exit(1)

        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(args.ssl_cert, args.ssl_key)
        protocol = "https"
        logger.info("SSL enabled - using HTTPS")

    # Run server
    logger.info(f"Starting server on {args.host}:{args.port}")
    logger.info("")
    logger.info("=" * 70)
    logger.info("Access the server at:")
    logger.info(f"  Local:   {protocol}://localhost:{args.port}")

    # Get network interfaces - try multiple methods for cross-platform support
    network_ips = []

    # Method 1: hostname -I (Linux)
    try:
        result = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=1)
        if result.returncode == 0:
            ips = result.stdout.strip().split()
            for ip in ips:
                # Filter out loopback and docker bridges (172.17.x.x)
                if not ip.startswith("127.") and not ip.startswith("172.17."):
                    network_ips.append(ip)
    except Exception:
        pass

    # Method 2: Socket method (cross-platform fallback)
    if not network_ips:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            if ip and ip != "127.0.0.1":
                network_ips.append(ip)
        except Exception:
            pass

    # Display all found network IPs
    for ip in network_ips:
        logger.info(f"  Network: {protocol}://{ip}:{args.port}")

    logger.info("=" * 70)
    logger.info("")
    logger.info("Press Ctrl+C to stop")

    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info("\nReceived signal to terminate. Shutting down gracefully...")
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")


def stop():
    """Stop the running process-vlm-webui server"""
    import sys
    import time

    try:
        import psutil
    except ImportError:
        logger.error("psutil is required for the stop command")
        logger.error("Install it with: pip install process-vlm-webui[dev]")
        sys.exit(1)

    print("Stopping Process VLM WebUI server...")

    # Find and kill processes running process_vlm_webui.server
    found = False
    killed = []

    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline")
            if cmdline:
                cmdline_str = " ".join(cmdline)
                if "process_vlm_webui.server" in cmdline_str or "process-vlm-webui" in cmdline_str:
                    # Don't kill the stop command itself
                    if "stop" not in cmdline_str:
                        found = True
                        print(f"  Stopping process {proc.info['pid']}: {proc.info['name']}")
                        proc.terminate()
                        killed.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    if not found:
        print("✓ No running server found")
        return

    # Wait for graceful shutdown
    time.sleep(2)

    # Force kill if still running
    for proc in killed:
        try:
            if proc.is_running():
                print(f"  Force killing process {proc.pid}")
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    # Final verification
    time.sleep(1)
    still_running = False
    for proc in psutil.process_iter(["cmdline"]):
        try:
            cmdline = proc.info.get("cmdline")
            if cmdline:
                cmdline_str = " ".join(cmdline)
                if "process_vlm_webui.server" in cmdline_str or "process-vlm-webui" in cmdline_str:
                    if "stop" not in cmdline_str:
                        still_running = True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    if still_running:
        print("❌ Failed to stop server")
        sys.exit(1)
    else:
        print("✓ Server stopped successfully")


if __name__ == "__main__":
    main()
