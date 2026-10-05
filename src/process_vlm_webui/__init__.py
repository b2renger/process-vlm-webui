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
Process VLM WebUI - Offline video analysis with Vision Language Models.

A web interface that takes an MP4 (or any OpenCV-readable video), samples it at
a configurable time step, describes every sampled frame with a Vision Language
Model, and durably logs each analysis to a transcript on disk.

Sibling project to live-vlm-webui, which does the same for a live camera feed.
"""

__version__ = "0.1.0"
__author__ = "NVIDIA Corporation"
__license__ = "Apache-2.0"

from . import server
from . import video_analyzer
from . import gpu_monitor
from . import vlm_service
from . import transcript

__all__ = ["server", "video_analyzer", "gpu_monitor", "vlm_service", "transcript"]
