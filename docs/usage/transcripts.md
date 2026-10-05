# Saving Analyses to Disk (Transcripts)

Every description the VLM produces is written to a timestamped transcript file
on the server. Recording is **on by default** — there is nothing to enable.

## Where the file goes

| Platform | Default location |
|----------|------------------|
| Linux / Jetson / DGX | `$XDG_DATA_HOME/process-vlm-webui/transcripts/` (usually `~/.local/share/process-vlm-webui/transcripts/`) |
| macOS | `~/Library/Application Support/process-vlm-webui/transcripts/` |
| Windows | `%APPDATA%\process-vlm-webui\transcripts\` |

Each server run creates its own file, `session-YYYYmmdd-HHMMSS.jsonl`, so runs
never interleave. The name is claimed atomically at startup, so several
instances started in the same second get `-1`, `-2` … suffixes rather than
sharing one file — see [Running Multiple Instances](./multiple-instances.md).
Backup snapshots land in a `backups/` subdirectory next to it.

```
~/.local/share/process-vlm-webui/transcripts/
├── session-20251217-142530.jsonl          ← live transcript
└── backups/
    ├── session-20251217-142530.20251217-142612.n25.jsonl
    └── session-20251217-142530.20251217-142809.n50.jsonl
```

## Record format

The default format is **JSONL** — one JSON object per line. Line-oriented
storage means a torn trailing line (from a power cut or `kill -9`) never
invalidates the records written before it.

```json
{"seq":42,"ts":"2026-09-21T14:26:03.118+02:00","ts_epoch":1789997163.118,"kind":"analysis","model":"gemma3:27b","api_base":"http://localhost:11434/v1","prompt":"Describe what you see in this image in one sentence.","response":"A person sitting at a desk in front of two monitors.","latency_ms":412.7,"ok":true,"source":"video:standup.mp4","job_id":"7e9bf8c75c93","video":"standup.mp4","step":21,"step_total":120,"frame_index":1260,"video_time_s":42.0,"timecode":"00:00:42.000","step_mode":"seconds","step_seconds":2.0,"frame_file":"00021_f1260.jpg"}
```

| Field | Meaning |
|-------|---------|
| `seq` | Record number within the file, starting at 1 |
| `ts` / `ts_epoch` | ISO-8601 local time with UTC offset, and Unix epoch seconds |
| `kind` | `analysis`, or a marker: `session_start`, `session_end`, `recording_paused`, `recording_resumed` |
| `model` / `api_base` | Which model produced the description, and where it ran |
| `prompt` | The exact prompt used — it can change mid-session from the WebUI |
| `response` | The model's text output |
| `latency_ms` | Inference time |
| `ok` / `error` | `false` plus an `error` string when the API call failed |
| `source` | `video:<filename>` |
| `video` | The filename you uploaded |
| `job_id` | Identifies one analysis run |
| `timecode` / `video_time_s` | Where in the video this frame came from |
| `frame_index` | The frame number in the source file |
| `step` / `step_total` | Position in the sampling plan |
| `step_mode` / `step_seconds` / `step_frames` | How the video was sampled |
| `frame_file` | The saved JPEG of this frame, under `frames/<job_id>/` |
| `frame` | Frame number the description was generated from |

Failed inferences are recorded too (`"ok": false`), so a gap in the timeline is
always explained rather than silently missing.

### Human-readable format

`--transcript-format txt` writes one line per record instead:

```
[2026-09-21T14:26:03.118+02:00] #42 <video:standup.mp4> gemma3:27b 413ms | A person sitting at a desk in front of two monitors.
```

## Backups

Every **25 records** (configurable) the live file is snapshotted into
`backups/`. The snapshot is written to a temporary file, fsync'd, then moved
into place with an atomic rename — an interrupted backup can never overwrite a
good snapshot with a partial one. The 10 most recent snapshots are kept and
older ones are pruned.

A final snapshot is always taken on graceful shutdown.

## Durability

- Each record is `flush()`ed and `fsync()`ed before the write call returns, so a
  hard kill loses at most the single in-flight description.
- Writes are serialized, so records land in the order they were produced.
- Disk I/O runs in a worker thread and never blocks the event loop.
- Any disk failure (permissions, full disk, unplugged drive) is counted and
  logged, then ignored — video streaming and inference keep running. The WebUI
  shows the error count so the failure is visible rather than silent.
- If the transcript path is unwritable at startup, recording disables itself and
  the server starts normally.

## Command-line options

```bash
process-vlm-webui --transcript-dir /mnt/data/vlm-logs --transcript-backup-every 50
```

| Option | Default | Description |
|--------|---------|-------------|
| `--no-transcript` | off | Disable recording entirely |
| `--transcript-dir PATH` | OS data dir | Directory for transcript files |
| `--transcript-file PATH` | — | Exact file path (overrides `--transcript-dir`); appends if it exists |
| `--transcript-format {jsonl,txt}` | `jsonl` | Structured or human-readable |
| `--transcript-backup-every N` | `25` | Snapshot every N records (`0` disables) |
| `--transcript-backup-keep K` | `10` | Snapshots to retain (`0` keeps all) |
| `--transcript-max-bytes BYTES` | `0` | Rotate the live file past this size (`0` disables) |
| `--no-transcript-fsync` | off | Skip fsync per record — faster, less durable |

Using `--transcript-file` with a fixed path appends across restarts, which is
useful for a single continuous log:

```bash
process-vlm-webui --transcript-file /var/log/vlm/continuous.jsonl --transcript-max-bytes 100000000
```

> [!WARNING]
> `--transcript-file` appends to exactly the path you give it and does **not**
> get a uniqueness suffix. Pointing two concurrent instances at the same file
> interleaves their records and duplicates sequence numbers. Give each instance
> its own path, or use `--transcript-dir` and let the auto-generated name keep
> them apart.

## WebUI

**Header pill** — a recording indicator sits in the top bar, left of the
connection status, so it is visible no matter where the sidebar is scrolled:

| State | Appearance | Meaning |
|-------|-----------|---------|
| `● REC 47` | red, pulsing dot | Recording; 47 records saved this session |
| `● PAUSED 47` | grey, static dot | Recording paused; the file is intact |
| `● REC 47` | amber | Recording, but writes are failing — check the panel |

**Click the pill to pause or resume recording.** The change takes effect on the
server immediately and is written into the transcript itself as a
`recording_paused` / `recording_resumed` marker, so the gap is explained in-file.

Pausing only stops recording — it does not stop video or VLM analysis. Recording
is independent of the Analyse/Stop buttons and stays in whatever state you
set it to.

**Sidebar panel** — the **Transcript** panel shows the live record count, backup
count, error count and target filename, with a toggle mirroring the header pill,
a **Backup** button to snapshot on demand, and a **Download** button to fetch the
file. Both controls stay in sync whichever one you use.

Both the pill and the panel are hidden when the server runs with `--no-transcript`.

## HTTP API

| Endpoint | Description |
|----------|-------------|
| `GET /api/transcript/status` | Writer state: path, counters, backup interval, last error |
| `GET /api/transcript/recent?limit=50` | Most recent records, newest last (max 500) |
| `GET /api/transcript/download` | Download the current transcript file |
| `POST /api/transcript/backup` | Force a snapshot now |
| `POST /api/transcript/config` | `{"enabled": bool, "backup_every": int}` |

```bash
curl http://localhost:8091/api/transcript/status
curl -X POST http://localhost:8091/api/transcript/backup
curl -X POST http://localhost:8091/api/transcript/config \
     -H 'Content-Type: application/json' -d '{"enabled": false}'
```

## Working with the data

```bash
# All descriptions, one per line
jq -r 'select(.kind == "analysis") | "\(.ts)  \(.response)"' session-*.jsonl

# Only failures
jq -r 'select(.ok == false) | "\(.ts)  \(.error)"' session-*.jsonl

# Average latency
jq -s 'map(select(.kind == "analysis") | .latency_ms) | add / length' session-*.jsonl

# Descriptions mentioning a person
jq -r 'select(.response? | test("person"; "i")) | "\(.ts)  \(.response)"' session-*.jsonl
```

```python
import json

with open("session-20251217-142530.jsonl") as f:
    records = [json.loads(line) for line in f if line.strip()]

analyses = [r for r in records if r["kind"] == "analysis"]
print(f"{len(analyses)} descriptions, {sum(not r['ok'] for r in analyses)} failed")
```
