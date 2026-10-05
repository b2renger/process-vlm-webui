# TODO / Roadmap

Ideas for Process VLM WebUI, roughly in order of how much they would add.

## Shipped in 0.1.0

- [x] Video file upload with server-side probing and rejection of undecodable files
- [x] Configurable time step (seconds or frames), range, frame cap, downscale width
- [x] Start / pause / resume / cancel over HTTP, with progress, rate and ETA
- [x] Live analysis timeline with per-frame thumbnails and click-to-seek
- [x] Transcript records carrying video timecode, frame index and sampling plan
- [x] Carry-context prompting for continuous narration
- [x] Reattach to a running job after a reload

## Next

- [ ] **Whole-video summary.** After a run, feed the descriptions back to the
      model as text and produce one paragraph for the clip. Cheap to add - the
      descriptions are already in `job.results` - and it is the most common
      thing people want after a timeline.
- [ ] **Scene-change sampling.** Instead of a fixed step, sample where the
      picture actually changes (frame differencing, or a histogram threshold).
      Fewer, better-placed frames for the same budget.
- [ ] **Batch queue.** Drop several files and have them analysed in turn.
      The job model already allows exactly one at a time; this is a queue in
      front of it plus a list in the UI.
- [ ] **Export formats.** WebVTT and SRT fall straight out of the timecodes and
      would make the output usable as subtitles or chapter markers.
- [ ] **Search within results.** Filter the timeline by text, and jump between
      matches.

## Maybe

- [ ] **Concurrent inference.** A remote API with capacity could run several
      frames at once. Would need care to keep transcript order intact - the
      ordering guarantee is worth more than the speed for most uses.
- [ ] **Audio transcription alongside the frames.** A parallel Whisper pass,
      merged into the same timeline by timecode.
- [ ] **Per-frame regions.** Crop to a region of interest before inference for
      fixed-camera footage.
- [ ] **Resume an interrupted job.** The transcript already records the plan, so
      a run that stopped at frame 240 of 400 could pick up from there.

## Known limitations

- One job at a time, by design. A second start returns `409`.
- Timecodes for strides above 60 frames rely on container seeking, which some
  files resolve to the nearest keyframe.
- Jobs and registered videos live in memory and do not survive a server restart;
  the transcript on disk does.
