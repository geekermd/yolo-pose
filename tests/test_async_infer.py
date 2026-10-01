"""Tests for the async inference worker that decouples display from the model."""

import time

import numpy as np
import pytest

from yolopose.vision.async_infer import InferenceWorker


def test_worker_produces_a_result():
    worker = InferenceWorker(lambda frame: int(frame.sum()))
    try:
        worker.submit(np.ones((4, 4), dtype=np.uint8))
        result = worker.wait_for_result(timeout=2.0)
        assert result == 16
        assert worker.stats.frames_inferred == 1
    finally:
        worker.close()


def test_worker_keeps_only_the_newest_frame():
    """A slow task must not build a backlog: stale frames are dropped."""
    release = []
    started = []

    def slow(frame):
        started.append(frame)
        time.sleep(0.25)
        return frame

    worker = InferenceWorker(slow)
    try:
        for index in range(6):
            worker.submit(np.full((2, 2), index, dtype=np.uint8))
        worker.wait_for_result(timeout=1.0)
        time.sleep(0.6)
        # Several frames arrived while the first was running, so some were dropped
        # rather than queued.
        assert worker.stats.frames_in == 6
        assert worker.stats.frames_dropped >= 3
    finally:
        worker.close()


def test_worker_survives_a_failing_task():
    def boom(frame):
        raise RuntimeError("model exploded")

    worker = InferenceWorker(boom)
    try:
        worker.submit(np.zeros((2, 2), dtype=np.uint8))
        time.sleep(0.3)
        # No result, but the worker must still be alive and accepting work.
        worker.submit(np.zeros((2, 2), dtype=np.uint8))
        time.sleep(0.3)
        assert worker._thread.is_alive()
    finally:
        worker.close()


def test_latest_and_freshness_tracking():
    worker = InferenceWorker(lambda frame: int(frame[0, 0]))
    try:
        worker.submit(np.full((2, 2), 7, dtype=np.uint8))
        worker.wait_for_result(timeout=2.0)
        value, serial = worker.latest()
        assert value == 7
        assert worker.is_fresh(serial - 1)
        assert not worker.is_fresh(serial)
    finally:
        worker.close()


def test_close_is_idempotent_and_stops_the_thread():
    worker = InferenceWorker(lambda frame: 1)
    worker.submit(np.zeros((2, 2), dtype=np.uint8))
    worker.wait_for_result(timeout=2.0)
    worker.close()
    worker.close()
    assert not worker._thread.is_alive()
    # Submitting after close must be a no-op rather than an error.
    worker.submit(np.zeros((2, 2), dtype=np.uint8))


def test_submit_after_close_does_not_raise():
    worker = InferenceWorker(lambda frame: 1)
    worker.close()
    worker.submit(np.zeros((2, 2), dtype=np.uint8))
    assert worker.stats.frames_in == 0
