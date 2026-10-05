# Changelog

All notable changes to Process VLM WebUI are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-21

First release. Forked from
[live-vlm-webui](https://github.com/nvidia-ai-iot/live-vlm-webui) 0.2.1 and
re-aimed at video files instead of a live camera feed.

### Added

- **Video file input.** Drag-and-drop or browse upload, streamed to disk in
  chunks so a multi-gigabyte file never has to fit in memory. Uploads are
  probed with OpenCV on arrival and rejected with a reason if undecodable.
- **Configurable time step.** Analyse one frame every N seconds of video, or
  every Nth frame. Start/end range, a cap on the number of frames, and a
  pre-inference downscale width are all configurable per job.
- **Analysis job control.** Start, pause, resume and cancel a run over HTTP;
  progress, throughput and ETA are reported throughout. One job at a time; a
  second start returns `409`.
- **Analysis timeline.** Every description streams into the UI over the
  existing WebSocket with its timecode and a thumbnail of the exact frame it
  came from. Clicking an entry seeks the player to that moment. The player can
  also follow the analysis as it runs.
- **Video context in the transcript.** Analysis records carry `video`,
  `timecode`, `video_time_s`, `frame_index`, `step`, `step_total` and
  `frame_file` alongside the fields the live app already wrote, and each run is
  bracketed by `video_job_start` / `video_job_end` records.
- **Carry-context prompting.** Optionally quote the previous description in the
  next prompt so the model describes a continuing scene rather than a series of
  unrelated stills.
- **Frame images on disk.** Each analysed frame is saved as a JPEG under
  `frames/<job-id>/`, served back to the UI and referenced from the transcript.
  Disable with `--no-save-frames`.
- **Reload recovery.** The UI reattaches to a job that is already running,
  rebuilding the timeline from `/api/video/results`.
- `--video PATH` to register a local file at startup, and
  `--allow-server-paths` to let clients analyse server-side paths (off by
  default, because the server binds `0.0.0.0`).

### Changed

- **No frame is ever dropped.** The live app skipped frames it could not keep
  up with; a file has no realtime deadline, so every planned frame is analysed
  and the caller waits. `VLMService.process_frame()` became
  `analyze_frame()`, which awaits instead of skipping.
- **HTTPS is opt-in.** With no camera permission to grant there is no reason to
  require TLS. Plain HTTP is the default; `--no-ssl` is gone and `--ssl` turns
  TLS on.
- Default port is 8091, so this can run alongside live-vlm-webui on 8090.
- Storage lives under `process-vlm-webui/` rather than `live-vlm-webui/`.
- The WebRTC settings section became timeline preferences (auto-scroll, follow
  the analysis, newest first).

### Fixed

- **`--model` is no longer silently overridden by the browser.** `/models`
  marked nothing as current whenever the UI supplied an `api_base` - which it
  always does - so the page auto-selected whichever model the backend listed
  first and pushed that to the server. The running model is now flagged when
  the request is about the endpoint already in use. (Inherited from upstream.)
- **Cancelling a job could crash the interpreter.** Cancelling only interrupts
  the `await`, never the decode running inside OpenCV, so releasing the
  `VideoCapture` from the teardown path could free it mid-read. All capture
  calls now run on one dedicated thread per job, which orders the release
  behind any in-flight read.
- The server's `--prompt` is now shown in the prompt editor, so the text on
  screen is the text that will be sent.

### Removed

- WebRTC, webcam and RTSP support, along with the `aiortc`, `av` and
  `aiohttp-cors` dependencies and the `rtsp_track.py` / `video_processor.py`
  modules.
- The mirror-video control and the max-frame-latency setting, both meaningless
  for a file.
