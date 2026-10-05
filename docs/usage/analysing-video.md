# Analysing a Video

How the sampling works, what each control does, and how to choose settings that
finish in the time you have.

## The pipeline

```
upload ──► probe ──► plan the sample ──► for each planned frame:
                                            seek/decode ──► downscale ──►
                                            VLM ──► transcript ──► UI
```

Everything after "plan" happens one frame at a time, in video order. There is
no realtime deadline, so unlike the live app **nothing is dropped**: every frame
in the plan is described, however slow the model is.

## Planning the sample

The plan is computed before a single frame is decoded, which is why the UI can
tell you the total and an ETA from the start.

```
stride  = round(step_seconds x fps)        # or step_frames directly
first   = round(start_s x fps)
last    = min(frame_count - 1, round(end_s x fps))
frames  = range(first, last + 1, stride)[:max_frames]
```

A few consequences worth knowing:

- **The stride is always at least 1 frame.** A time step shorter than one frame
  interval does not sample the same frame twice; it samples every frame.
- **Sub-second steps work.** `0.5` on a 30 fps source is every 15th frame.
- **A container that reports no frame count** (a truncated or streamed file)
  falls back to its duration. If the file ends earlier than its header claims,
  the job stops there and reports what it did.
- **An implausible frame rate** (0, or above 1000) is replaced with 30 fps and
  logged, so a broken header cannot make the time step meaningless.

## Decoding

Two strategies, chosen per frame by how far ahead the next sample is:

- **Gap of 60 frames or less:** walk forward with `grab()`, which advances the
  demuxer without decoding, and only decode the frame actually wanted. Accurate
  and cheap for small strides.
- **Larger gap:** seek by frame index. Much faster over a long video, at the
  cost of landing on whatever the container's index gives back.

All capture calls for a job run on one dedicated thread. That is what makes
cancelling safe: cancelling interrupts the `await`, never the decode already
running inside OpenCV, so the release is ordered behind any in-flight read
rather than freeing a capture that is still being used.

## The controls

| Control | Flag | Effect |
| --- | --- | --- |
| Time step | `--step` / `--step-frames` | Seconds (or frames) between analysed frames |
| Start / End | - | Restrict to part of the video |
| Max frames | - | Hard cap on descriptions, applied after the range |
| Frame width | `--max-width` | Downscale before inference; `0` keeps the original |
| Carry previous description | `--carry-context` | See below |

### Carry previous description

Off by default. Each frame is described on its own, which is what you want for
search, indexing or per-moment labelling - the descriptions stay independent
and comparable.

Turn it on and the previous description is quoted in the next prompt:

```
<your prompt>

For continuity, the previous sampled moment of this video was described as:
"<previous description>"
Describe the current frame (at 00:01:24.000) on its own terms, noting what has
changed only when it matters.
```

That produces something closer to a narration of a continuing scene. The exact
prompt used is what gets written to the transcript, so a run is always
reconstructable.

Two things to know: an error in one description propagates into the next
prompt's context, and the prompts get longer, so tokens cost more.

## Choosing a step

Wall-clock is roughly:

```
duration / step  x  model latency
```

| Video | Step | Frames | At 2 s/frame |
| --- | --- | --- | --- |
| 5 min | 1 s | 300 | 10 min |
| 5 min | 5 s | 60 | 2 min |
| 1 hour | 10 s | 360 | 12 min |
| 1 hour | 30 s | 120 | 4 min |

The step to pick is the one that matches how fast the content changes, not the
one that produces the most rows. A fixed-camera scene barely changes in 10
seconds; a hand-held walkthrough changes in 2.

`--max-width` is the other lever, and usually the cheaper one: halving the width
quarters the pixels, which for most VLMs roughly quarters the image tokens.

## Running it headless

The UI is optional. Upload, start, and read the transcript:

```bash
BASE=http://localhost:8091

VIDEO=$(curl -s -F "file=@clip.mp4" $BASE/api/video/upload \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["video"]["video_id"])')

curl -s -X POST $BASE/api/video/start -H 'Content-Type: application/json' \
     -d "{\"video_id\":\"$VIDEO\",\"step_seconds\":5,\"max_width\":768,
          \"prompt\":\"Describe the action in one sentence.\"}"

# Wait for it
until curl -s $BASE/api/video/status | grep -q '"active": false'; do sleep 5; done

curl -s $BASE/api/video/results \
  | python3 -c 'import json,sys; [print(r["timecode"], r["text"]) for r in json.load(sys.stdin)["results"]]'
```

For a file already on the server, start with `--allow-server-paths` and use
`/api/video/register` with a `path` instead of uploading.

## Reattaching

A job lives on the server, not in the page. Close the tab, reload, or open a
second browser - the UI asks `/api/video/status`, rebuilds the timeline from
`/api/video/results`, and picks the progress up where it is.

Only one job runs at a time. Starting a second returns `409` with the running
job's status, rather than queueing it.

## See also

- [Transcripts](transcripts.md) - the on-disk format, backups and the API
- [Advanced configuration](advanced-configuration.md) - every CLI flag
- [GPU monitoring](gpu-monitoring.md) - reading the system stats panel
- [List of VLMs](list-of-vlms.md) - models known to work
