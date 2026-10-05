# Documentation Images

Screenshots of Process VLM WebUI, captured from a real run against a local
Ollama backend.

| File | What it shows |
| --- | --- |
| `app-light-theme.png` | The full app after a finished run, light theme |
| `app-dark-theme.png` | The same, dark theme |
| `video-controls.png` | The video source and sampling panel on its own |

## Regenerating them

They are produced by driving the real app with Playwright, so they never drift
from what the UI actually looks like:

1. Start a server against a working VLM backend.
2. Load a short clip and run an analysis.
3. Screenshot with `page.screenshot(full_page=True)` at
   `device_scale_factor=2`, viewport 1560x1150.

Reference one in markdown with `![Alt text](docs/images/filename.png)`.
