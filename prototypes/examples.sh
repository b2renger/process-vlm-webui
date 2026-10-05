#!/bin/bash
# Example invocations of Process VLM WebUI.
#
# Everything here is commented out - copy the line you want.
# The prompt applies to every sampled frame of the video.

# ============================================================================
# Scene description, one line per 5 seconds of video
# ============================================================================
# process-vlm-webui --model gemma3:27b --api-base http://localhost:11434/v1 \
#   --step 5 \
#   --prompt "Describe what you see in this image in one sentence."

# ============================================================================
# Narration - each description knows what came before it
# ============================================================================
# process-vlm-webui --model gemma3:27b --api-base http://localhost:11434/v1 \
#   --step 3 --carry-context \
#   --prompt "Describe this moment of the video in one or two sentences."

# ============================================================================
# Action log - dense sampling, short answers
# ============================================================================
# process-vlm-webui --model gemma3:27b --api-base http://localhost:11434/v1 \
#   --step 1 --max-width 640 \
#   --prompt "Describe the action taking place. If nothing is happening, say 'No activity'."

# ============================================================================
# Safety review of site footage
# ============================================================================
# process-vlm-webui --model gemma3:27b --api-base http://localhost:11434/v1 \
#   --step 10 \
#   --prompt "Are there any safety hazards visible? Answer with 'ALERT: <description>' or 'SAFE'."

# ============================================================================
# Searchable index - one row per shot, objects only
# ============================================================================
# process-vlm-webui --model gemma3:27b --api-base http://localhost:11434/v1 \
#   --step 2 --transcript-format jsonl \
#   --prompt "List every distinct object and person visible, separated by commas. No other text."

# ============================================================================
# OCR pass over a screen recording
# ============================================================================
# process-vlm-webui --model gemma3:27b --api-base http://localhost:11434/v1 \
#   --step 4 --max-width 1920 \
#   --prompt "Transcribe all text visible in this frame. If there is none, say 'No text'."

# ============================================================================
# Long video, fast pass - coarse step and small frames
# ============================================================================
# process-vlm-webui --model gemma3:27b --api-base http://localhost:11434/v1 \
#   --step 30 --max-width 512 \
#   --prompt "Summarise this frame in under ten words."

# ============================================================================
# Preload a file and skip the upload
# ============================================================================
# process-vlm-webui --video ~/footage/survey.mp4 --step 5

# ============================================================================
# Accessibility - detailed description of each moment
# ============================================================================
# process-vlm-webui --model gemma3:27b --api-base http://localhost:11434/v1 \
#   --step 8 \
#   --prompt "Provide a detailed description of the scene for a visually impaired person."
