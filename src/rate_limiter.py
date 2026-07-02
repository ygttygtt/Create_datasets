"""Rate limiter using asyncio semaphore + sliding-window RPM control."""

import asyncio
import time
from collections import deque


class RateLimiter:
    """Controls API request concurrency and rate."""

    def __init__(
        self,
        max_concurrent: int = 2,
        delay_between_calls: float = 1.0,
        requests_per_minute: int = 30,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ):
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._delay = delay_between_calls
        self._rpm = requests_per_minute
        self._max_retries = max_retries
        self._retry_delay = retry_delay

        # Sliding window for RPM tracking
        self._call_times: deque[float] = deque()
        self._last_call_time: float = 0.0

    async def acquire(self) -> None:
        """Wait until a request slot is available."""
        # Enforce inter-call delay
        now = time.monotonic()
        wait = self._delay - (now - self._last_call_time)
        if wait > 0:
            await asyncio.sleep(wait)

        # Enforce RPM cap
        if self._rpm > 0:
            now = time.monotonic()
            # Remove calls older than 60s
            while self._call_times and now - self._call_times[0] > 60:
                self._call_times.popleft()
            if len(self._call_times) >= self._rpm:
                sleep_time = 60 - (now - self._call_times[0]) + 0.1
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

        await self._semaphore.acquire()
        self._last_call_time = time.monotonic()
        if self._rpm > 0:
            self._call_times.append(self._last_call_time)

    def release(self) -> None:
        """Release a request slot."""
        self._semaphore.release()

    def retry_delays(self) -> list[float]:
        """Exponential backoff delays for retries."""
        return [self._retry_delay * (2 ** i) for i in range(self._max_retries)]
