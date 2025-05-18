import time
import random
from datetime import datetime, timedelta
from collections import deque
import logging
from functools import wraps
from google.genai.errors import ClientError

logger = logging.getLogger(__name__)

class RateLimitHandler:
    def __init__(self, calls_per_minute=10, max_retries=3, initial_delay=1.0):
        self.calls_per_minute = calls_per_minute
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.call_times = deque(maxlen=calls_per_minute)
        
    def _wait_if_needed(self):
        """Check and wait if we're hitting rate limits"""
        now = datetime.now()
        while self.call_times and (now - self.call_times[0]) > timedelta(minutes=1):
            self.call_times.popleft()
            
        if len(self.call_times) >= self.calls_per_minute:
            wait_time = 60 - (now - self.call_times[0]).total_seconds()
            if wait_time > 0:
                logger.info(f"Rate limit reached. Waiting {wait_time:.2f}s")
                time.sleep(wait_time)
        
        self.call_times.append(now)

    def with_retries(self, func, *args, **kwargs):
        """Execute function with retries and exponential backoff"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                self._wait_if_needed()
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                    delay = self.initial_delay * (2 ** attempt) + random.random()
                    logger.warning(f"Rate limit exceeded. Retrying in {delay:.2f}s (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(delay)
                    continue
                raise
                
        logger.error(f"Failed after {self.max_retries} retries: {last_error}")
        raise last_error
