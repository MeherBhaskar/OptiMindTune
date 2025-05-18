import asyncio
import logging
import re
from functools import wraps
from google.genai.errors import ClientError
import time

logger = logging.getLogger(__name__)

class RateLimitHandler:
    def __init__(self, base_delay=2.0, max_retries=3):
        self.base_delay = base_delay
        self.max_retries = max_retries
        self._last_call = 0
        self.min_interval = 5  # Minimum seconds between calls

    def _extract_retry_delay(self, error_message: str) -> int:
        if match := re.search(r"retryDelay': '(\d+)s'", str(error_message)):
            return int(match.group(1))
        return self.base_delay

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries <= self.max_retries:
                try:
                    # Enforce minimum interval between calls
                    now = time.time()
                    time_since_last = now - self._last_call
                    if time_since_last < self.min_interval:
                        time.sleep(self.min_interval - time_since_last)
                    
                    self._last_call = time.time()
                    return func(*args, **kwargs)

                except ClientError as e:
                    if "RESOURCE_EXHAUSTED" in str(e):
                        retries += 1
                        if retries > self.max_retries:
                            raise

                        delay = self._extract_retry_delay(str(e))
                        logger.warning(f"Rate limit exceeded. Waiting {delay}s before retry {retries}/{self.max_retries}")
                        time.sleep(delay)
                    else:
                        raise
            return None
        return wrapper
