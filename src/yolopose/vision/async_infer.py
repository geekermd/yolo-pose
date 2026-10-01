"""Asynchronous inference so the preview never waits for the model.

The display and the model are decoupled: frames are pushed into a single-slot
queue and a worker thread keeps the newest detection result up to date.  The
render loop then always draws the freshest frame at the camera's own rate, with
whatever overlay is currently available.

On a CPU-bound machine this is what turns "display waits for inference" into
"display runs at the camera's frame rate".  The overlay lags the picture by one
inference latency, which is the correct trade for a live preview.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple

import numpy as np

__all__ = ["InferenceWorker", "WorkerStats"]


@dataclass
class WorkerStats:
    """Counters describing what the worker has been doing."""

    frames_in: int = 0
    frames_inferred: int = 0
    frames_dropped: int = 0
    last_latency: float = 0.0
    avg_latency: float = 0.0

    @property
    def drop_rate(self) -> float:
        total = self.frames_inferred + self.frames_dropped
        return (self.frames_dropped / total) if total else 0.0


class InferenceWorker:
    """Runs a callable over the newest frame in a background thread.

    Parameters
    ----------
    task:
        ``task(frame) -> result``.  Called on the worker thread.
    keep_latest_only:
        Drop frames that arrive while an inference is running, so the worker
        never falls behind the camera.  This is what keeps the preview live.
    """

    def __init__(self, task: Callable[[np.ndarray], Any],
                 keep_latest_only: bool = True, name: str = "yolo-pose-infer"):
        self._task = task
        self._keep_latest_only = keep_latest_only
        self._condition = threading.Condition()
        self._slot: Optional[np.ndarray] = None
        self._result: Optional[Any] = None
        self._result_serial: int = 0
        self._consumed_serial: int = 0
        self._submitted: int = 0
        self._stop = False
        self.stats = WorkerStats()
        self._latencies: list = []
        self._thread = threading.Thread(target=self._loop, name=name, daemon=True)
        self._thread.start()

    # -- producer side ------------------------------------------------------ #
    def submit(self, frame: np.ndarray) -> None:
        """Hand a frame to the worker; replaces any frame not yet picked up."""
        with self._condition:
            if self._stop:
                return
            self.stats.frames_in += 1
            if self._slot is not None:
                self.stats.frames_dropped += 1
                if not self._keep_latest_only:
                    # Still replace: for a live view the newest frame wins.
                    pass
            self._slot = frame
            self._submitted += 1
            self._condition.notify()

    # -- consumer side ------------------------------------------------------ #
    def latest(self) -> Tuple[Optional[Any], int]:
        """Newest result and its serial, without blocking."""
        with self._condition:
            return self._result, self._result_serial

    def is_fresh(self, serial: int) -> bool:
        """True when a result newer than ``serial`` is available."""
        with self._condition:
            return self._result_serial > serial

    def wait_for_result(self, timeout: float = 1.0) -> Optional[Any]:
        """Block until at least one result exists (used to prime the first frame)."""
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._result is None and not self._stop:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
            return self._result

    # -- worker ------------------------------------------------------------- #
    def _loop(self) -> None:
        while True:
            with self._condition:
                while self._slot is None and not self._stop:
                    self._condition.wait(0.1)
                if self._stop:
                    return
                frame = self._slot
                self._slot = None

            started = time.perf_counter()
            try:
                result = self._task(frame)
            except Exception:
                # A failing task must not kill the preview loop.
                result = None
            elapsed = time.perf_counter() - started

            with self._condition:
                if result is not None:
                    self._result = result
                    self._result_serial += 1
                    self.stats.frames_inferred += 1
                    self._latencies.append(elapsed)
                    if len(self._latencies) > 30:
                        self._latencies.pop(0)
                    self.stats.last_latency = elapsed
                    self.stats.avg_latency = sum(self._latencies) / len(self._latencies)
                self._condition.notify_all()

    # -- lifecycle ---------------------------------------------------------- #
    def close(self, timeout: float = 2.0) -> None:
        with self._condition:
            self._stop = True
            self._condition.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def __enter__(self) -> "InferenceWorker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
