import asyncio
import logging
from functools import wraps
from google.genai.errors import ClientError
import re
from typing import Optional

logger = logging.getLogger(__name__)

class RateLimitHandler:
    def __init__(self, base_delay: float = 1.0, max_retries: int = 3):
        self.base_delay = base_delay
        self.max_retries = max_retries

    @staticmethod
    def extract_retry_delay(error_message: str) -> Optional[float]:
        """Extract retry delay from error message"""
        if match := re.search(r"retryDelay': '(\d+)s'", str(error_message)):
            return float(match.group(1))
        return None

    def __call__(self, func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retries = 0
            while retries <= self.max_retries:
                try:
                    return await func(*args, **kwargs)
                except ClientError as e:
                    if e.status_code == 429:  # Rate limit exceeded
                        retries += 1
                        if retries > self.max_retries:
                            raise  # Max retries exceeded

                        # Calculate delay with exponential backoff
                        retry_delay = self.extract_retry_delay(str(e)) or (self.base_delay * (2 ** (retries - 1)))
                        logger.warning(f"Rate limit exceeded. Retrying in {retry_delay} seconds... (Attempt {retries}/{self.max_retries})")
                        await asyncio.sleep(retry_delay)
                    else:
                        raise
            return await func(*args, **kwargs)
        return wrapper
