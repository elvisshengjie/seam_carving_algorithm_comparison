"""Optional cooperative cancellation and throttled progress for long computations."""

from time import perf_counter
from typing import Callable


class ComputationCancelled(Exception):
    """The caller requested cancellation; no completed resize result is returned."""


class RunControl:
    def __init__(self, should_cancel: Callable[[], bool] | None = None,
                 on_status: Callable[[str], None] | None = None,
                 min_interval: float = 0.15):
        self.should_cancel = should_cancel
        self.on_status = on_status
        self.min_interval = min_interval
        self.last_status = float("-inf")

    def checkpoint(self, message: str | None = None) -> None:
        if self.should_cancel is not None and self.should_cancel():
            raise ComputationCancelled()
        if message is not None and self.on_status is not None:
            now = perf_counter()
            if now - self.last_status >= self.min_interval:
                self.last_status = now
                self.on_status(message)
        if self.should_cancel is not None and self.should_cancel():
            raise ComputationCancelled()
