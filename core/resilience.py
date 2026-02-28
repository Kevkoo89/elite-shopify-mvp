from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar

from .logging import get_logger

T = TypeVar("T")
logger = get_logger("core.resilience")


def with_retry(
    action: Callable[[], T],
    *,
    retries: int = 3,
    base_delay_s: float = 0.7,
    max_delay_s: float = 5.0,
    jitter_ratio: float = 0.2,
    action_name: str = "action",
) -> T:
    last_exc: Exception | None = None
    bounded_retries = max(1, min(8, retries))
    bounded_base_delay = max(0.0, min(3.0, base_delay_s))
    bounded_max_delay = max(0.1, min(15.0, max_delay_s))
    bounded_jitter = max(0.0, min(0.5, jitter_ratio))

    for attempt in range(1, bounded_retries + 1):
        try:
            return action()
        except Exception as exc:  # noqa: PERF203
            last_exc = exc
            logger.warning("retry %s/%s for %s failed", attempt, bounded_retries, action_name)
            if attempt < bounded_retries:
                delay = min(bounded_max_delay, bounded_base_delay * (2 ** (attempt - 1)))
                time.sleep(delay + delay * bounded_jitter * random.random())

    assert last_exc is not None
    raise last_exc
