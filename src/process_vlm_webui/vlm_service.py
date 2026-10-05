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
VLM Service
Handles async image analysis using any OpenAI-compatible VLM API
(Works with vLLM, SGLang, Ollama, OpenAI, etc.)

Frames come from a video file rather than a live camera, so every frame handed
to analyze_frame() is analysed - nothing is dropped to keep up with realtime.
"""

import asyncio
import base64
import io
import time
from openai import AsyncOpenAI
from PIL import Image
from typing import Any, Optional
import logging

from .transcript import TranscriptWriter

logger = logging.getLogger(__name__)


class VLMService:
    """Service for analyzing images using VLM via OpenAI-compatible API"""

    def __init__(
        self,
        model: str,
        api_base: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        prompt: str = "Describe what you see in this image in one sentence.",
        max_tokens: int = 512,
        transcript: Optional[TranscriptWriter] = None,
    ):
        """
        Initialize VLM service

        Args:
            model: Model name (e.g., "llama-3.2-11b-vision-instruct" for vLLM)
            api_base: Base URL for the API (e.g., "http://localhost:8000/v1" for vLLM)
            api_key: API key (use "EMPTY" for local servers)
            prompt: Default prompt to use for image analysis
            max_tokens: Maximum tokens to generate
            transcript: Optional TranscriptWriter that persists every analysis
                to disk. Failures there never affect inference.
        """
        self.model = model
        self.api_base = api_base
        self.api_key = api_key if api_key else "EMPTY"
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.client = AsyncOpenAI(base_url=api_base, api_key=api_key)
        self.transcript = transcript
        self.current_response = "Initializing..."
        self.is_processing = False
        self._processing_lock = asyncio.Lock()

        # Metrics tracking
        self.last_inference_time = 0.0  # seconds
        self.total_inferences = 0
        self.total_inference_time = 0.0

    async def analyze_image(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Analyze an image using the VLM model

        Args:
            image: PIL Image to analyze
            prompt: Prompt for the VLM (uses default if None)
            context: Optional metadata (e.g. source, frame number) attached to
                the transcript record for this analysis

        Returns:
            Generated response string
        """
        if prompt is None:
            prompt = self.prompt

        # Started outside the try so the error path can still report latency
        start_time = time.perf_counter()

        try:
            # Convert PIL Image to base64
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format="JPEG")
            img_byte_arr = img_byte_arr.getvalue()
            img_base64 = base64.b64encode(img_byte_arr).decode("utf-8")

            # Create message with image
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                        },
                    ],
                }
            ]

            # Call API
            response = await self.client.chat.completions.create(
                model=self.model, messages=messages, max_tokens=self.max_tokens, temperature=0.7
            )

            # Calculate latency
            end_time = time.perf_counter()
            inference_time = end_time - start_time

            # Update metrics
            self.last_inference_time = inference_time
            self.total_inferences += 1
            self.total_inference_time += inference_time

            result = response.choices[0].message.content.strip()
            logger.info(f"VLM response: {result} (latency: {inference_time*1000:.0f}ms)")

            # Persist before returning so a description is on disk before the
            # next inference starts. Transcript errors never reach the caller.
            await self._record(prompt, result, inference_time, context, ok=True)
            return result

        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            message = f"Error: {str(e)}"
            # Record failures too, so gaps in the transcript are explained
            await self._record(
                prompt,
                message,
                time.perf_counter() - start_time,
                context,
                ok=False,
                error=str(e),
            )
            return message

    async def _record(
        self,
        prompt: str,
        response: str,
        latency: float,
        context: Optional[dict[str, Any]] = None,
        ok: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """
        Write one analysis to the transcript. Never raises.

        TranscriptWriter.write() already swallows I/O errors; the extra guard
        here means even a programming error in the transcript layer cannot
        break the inference path.
        """
        if self.transcript is None:
            return

        record = {
            "model": self.model,
            "api_base": self.api_base,
            "prompt": prompt,
            "response": response,
            "latency_ms": round(latency * 1000, 1),
            "ok": ok,
        }
        if error:
            record["error"] = error
        if context:
            record.update(context)

        try:
            await self.transcript.write(record)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"Unexpected transcript failure (analysis not saved): {e}")

    async def analyze_frame(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Analyze one sampled frame, waiting for any in-flight inference first.

        The live app dropped frames it could not keep up with; a video file has
        no such deadline, so here every planned frame is analysed and the caller
        waits. The lock still serializes inference, which keeps descriptions -
        and therefore the transcript - in strict video time order.

        Args:
            image: PIL Image to process
            prompt: Optional custom prompt (uses default if None)
            context: Optional metadata (video, timecode, frame index) for the
                transcript

        Returns:
            The model's description, or an "Error: ..." string on failure
        """
        async with self._processing_lock:
            self.is_processing = True
            try:
                response = await self.analyze_image(image, prompt, context)
                self.current_response = response
                return response
            finally:
                self.is_processing = False

    def get_current_response(self) -> tuple[str, bool]:
        """
        Get the current response and processing status

        Returns:
            Tuple of (response, is_processing)
        """
        return self.current_response, self.is_processing

    def get_metrics(self) -> dict:
        """
        Get current performance metrics

        Returns:
            Dict with latency and throughput metrics
        """
        avg_latency = (
            self.total_inference_time / self.total_inferences if self.total_inferences > 0 else 0.0
        )

        metrics = {
            "last_latency_ms": self.last_inference_time * 1000,
            "avg_latency_ms": avg_latency * 1000,
            "total_inferences": self.total_inferences,
            "is_processing": self.is_processing,
        }

        # Piggyback compact transcript state on the existing per-frame broadcast
        if self.transcript is not None:
            metrics["transcript"] = self.transcript.brief_stats()

        return metrics

    def update_prompt(self, new_prompt: str, max_tokens: Optional[int] = None) -> None:
        """
        Update the default prompt and optionally max_tokens

        Args:
            new_prompt: New prompt to use
            max_tokens: Maximum tokens to generate (optional)
        """
        self.prompt = new_prompt
        if max_tokens is not None:
            self.max_tokens = max_tokens
            logger.info(f"Updated prompt to: {new_prompt}, max_tokens: {max_tokens}")
        else:
            logger.info(f"Updated prompt to: {new_prompt}")

    def update_api_settings(
        self, api_base: Optional[str] = None, api_key: Optional[str] = None
    ) -> None:
        """
        Update API base URL and/or API key, recreating the client

        Args:
            api_base: New API base URL (optional)
            api_key: New API key (optional, use empty string for local services)
        """
        if api_base:
            self.api_base = api_base
        if api_key is not None:  # Allow empty string
            self.api_key = api_key if api_key else "EMPTY"

        # Recreate the client with new settings
        self.client = AsyncOpenAI(base_url=self.api_base, api_key=self.api_key)

        masked_key = (
            "***" + self.api_key[-4:]
            if self.api_key and len(self.api_key) > 4 and self.api_key != "EMPTY"
            else "EMPTY"
        )
        logger.info(f"Updated API settings - base: {self.api_base}, key: {masked_key}")
