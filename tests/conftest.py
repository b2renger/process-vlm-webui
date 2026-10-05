"""Shared pytest fixtures and configuration for all tests."""

import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock

from tests.utils.performance import PerformanceMetrics

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "fixtures" / "data"


# Register custom markers
def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line("markers", "performance: mark test as a performance test (can be slow)")
    config.addinivalue_line(
        "markers", "slow: mark test as slow running (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "e2e: mark test as end-to-end test")


# Note: We don't define event_loop fixture here because:
# 1. pytest-asyncio will create one automatically with asyncio_mode="strict"
# 2. AioHTTPTestCase manages its own event loop and doesn't use pytest-asyncio


@pytest.fixture
def mock_vlm_service():
    """Mock VLM service for testing without actual model calls."""
    service = AsyncMock()
    service.query_async.return_value = {
        "response": "Test response from VLM",
        "model": "test-model",
    }
    service.is_ready.return_value = True
    return service


@pytest.fixture
def mock_gpu_monitor():
    """Mock GPU monitor for testing without actual GPU."""
    monitor = Mock()
    monitor.get_stats.return_value = {
        "gpu_utilization": 45,
        "memory_used": 2048,
        "memory_total": 8192,
        "temperature": 65,
    }
    return monitor


@pytest.fixture
def sample_image_path():
    """Path to a sample test image."""
    return TEST_DATA_DIR / "sample_image.jpg"


@pytest.fixture
def sample_video_path():
    """Path to a sample test video."""
    return TEST_DATA_DIR / "sample_video.mp4"


@pytest.fixture
async def test_server():
    """Create a test server instance."""
    from process_vlm_webui.server import create_app

    app = await create_app(test_mode=True)

    # You may need to adjust this based on your server implementation
    yield app

    # Cleanup
    await app.cleanup()


@pytest.fixture
def mock_video_analyzer():
    """Mock video analysis job for testing without decoding a real file."""
    # status() is synchronous on the real job, so a plain Mock is the right
    # stand-in here - an AsyncMock would hand the test a coroutine instead
    job = Mock()
    job.cancel = AsyncMock()
    job.status.return_value = {
        "state": "running",
        "analyzed": 1,
        "total": 10,
        "active": True,
    }
    return job


@pytest.fixture
def tiny_video(tmp_path):
    """
    Write a short, real video file and return (path, fps, frame_count).

    Generating it here rather than committing a binary keeps the fixture
    self-describing: the frame index is painted into each frame, so a test can
    assert that the analyzer sampled the frames it claims to have sampled.
    """
    import cv2
    import numpy as np

    fps, frames, width, height = 10, 50, 160, 120
    path = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    assert writer.isOpened(), "OpenCV could not open an mp4 writer"
    for i in range(frames):
        frame = np.full((height, width, 3), i * 5 % 256, dtype=np.uint8)
        cv2.putText(frame, str(i), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        writer.write(frame)
    writer.release()
    return path, fps, frames


@pytest.fixture
def performance_metrics():
    """Provide a PerformanceMetrics instance for tests."""
    return PerformanceMetrics()


@pytest.fixture(scope="session")
def performance_report(request):
    """Collect performance metrics across all tests and report at end."""
    metrics = PerformanceMetrics()

    yield metrics

    # Print summary at end of test session
    if metrics.timings:
        print("\n" + "=" * 70)
        print("PERFORMANCE TEST SUMMARY")
        print("=" * 70)
        print(metrics.summary())
