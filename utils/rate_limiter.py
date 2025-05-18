import time
import random
from datetime import datetime, timedelta
from collections import deque
import logging
from functools import wraps
from google.genai.errors import ClientError

logger = logging.getLogger(__name__)

class RateLimitHandler:
    def __init__(self, calls_per_minute=10, max_retries=3, initial_delay=1.0, timeout=60):
        self.calls_per_minute = calls_per_minute
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.timeout = timeout  # maximum time to wait for a single request (seconds)
        self.call_times = deque(maxlen=calls_per_minute)
        
    def _wait_if_needed(self):
        now = datetime.now()
        while self.call_times and (now - self.call_times[0]) > timedelta(minutes=1):
            self.call_times.popleft()
        if len(self.call_times) >= self.calls_per_minute:
            wait_time = 60 - (now - self.call_times[0]).total_seconds()
            if wait_time > 0:
                logger.info(f"Rate limit reached. Waiting {wait_time:.2f}s")
                time.sleep(wait_time)
        self.call_times.append(now)

    def _extract_retry_delay(self, error):
        try:
            if hasattr(error, "args") and error.args:
                msg = str(error.args[0])
                import re
                match = re.search(r"retryDelay['\"]?: ['\"]?(\d+)s['\"]?", msg)
                if match:
                    return int(match.group(1))
            if hasattr(error, "args") and len(error.args) > 1:
                details = error.args[1]
                if isinstance(details, dict):
                    for d in details.get("error", {}).get("details", []):
                        if "@type" in d and "RetryInfo" in d["@type"]:
                            retry_delay = d.get("retryDelay")
                            if retry_delay and retry_delay.endswith("s"):
                                return int(retry_delay.rstrip("s"))
        except Exception:
            pass
        return None

    def with_retries(self, func, *args, **kwargs):
        last_error = None
        for attempt in range(self.max_retries):
            try:
                self._wait_if_needed()
                start_time = time.time()
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                last_error = e
                retry_delay = self._extract_retry_delay(e)
                if retry_delay:
                    logger.warning(f"Rate limit exceeded. Retrying in {retry_delay:.2f}s (server suggested).")
                    if retry_delay > self.timeout:
                        logger.error(f"Retry delay {retry_delay}s exceeds timeout {self.timeout}s. Aborting.")
                        break
                    time.sleep(retry_delay)
                    continue
                if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                    delay = min(self.initial_delay * (2 ** attempt) + random.random(), self.timeout)
                    logger.warning(f"Rate limit exceeded. Retrying in {delay:.2f}s (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(delay)
                    continue
                raise
        logger.error(f"Failed after {self.max_retries} retries: {last_error}")
        raise last_error
