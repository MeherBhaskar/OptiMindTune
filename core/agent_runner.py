"""
Agent Runner: Handles agent execution with rate limiting and error handling
"""

import logging
from typing import Optional
from google.adk.runners import Runner
from google.genai import types
from utils.rate_limiter import RateLimitHandler
from config import OptiMindConfig

logger = logging.getLogger(__name__)


class AgentRunner:
    """Manages agent execution with rate limiting and error handling."""
    
    def __init__(self, config: OptiMindConfig):
        self.config = config
        self.rate_limiter = RateLimitHandler(
            calls_per_minute=10,
            max_retries=config.max_retries,
            initial_delay=config.rate_limit_delay
        )
    
    def run_agent(self, runner: Runner, message: types.Content, session_id: str) -> Optional[any]:
        """
        Run an agent with proper session ID and rate limiting.
        
        Args:
            runner: The agent runner instance
            message: Message to send to the agent
            session_id: Session ID for this interaction
            
        Returns:
            Agent response event or None if failed
        """
        def _execute_agent():
            """Internal function to execute the agent."""
            try:
                for event in runner.run(
                    user_id=self.config.user_id,
                    session_id=session_id,
                    new_message=message
                ):
                    if event.is_final_response():
                        return event
                return None
            except Exception as e:
                logger.error(f"Agent execution failed: {e}")
                raise
        
        try:
            # Execute with rate limiting and retries
            result = self.rate_limiter.with_retries(_execute_agent)
            
            if result is None:
                logger.warning(f"Agent returned no final response for session {session_id}")
            else:
                logger.debug(f"Agent completed successfully for session {session_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Agent run failed after retries for session {session_id}: {e}")
            return None
    
    def run_agent_with_timeout(self, runner: Runner, message: types.Content, 
                              session_id: str, timeout: float = 30.0) -> Optional[any]:
        """
        Run an agent with a timeout.
        
        Args:
            runner: The agent runner instance
            message: Message to send to the agent
            session_id: Session ID for this interaction
            timeout: Timeout in seconds
            
        Returns:
            Agent response event or None if failed/timeout
        """
        import signal
        import contextlib
        
        class TimeoutError(Exception):
            pass
        
        @contextlib.contextmanager
        def timeout_context(seconds):
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Agent execution timed out after {seconds} seconds")
            
            # Set the signal handler
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(seconds))
            
            try:
                yield
            finally:
                # Restore the old signal handler
                signal.signal(signal.SIGALRM, old_handler)
                signal.alarm(0)
        
        try:
            with timeout_context(timeout):
                return self.run_agent(runner, message, session_id)
        except TimeoutError as e:
            logger.error(f"Agent execution timed out: {e}")
            return None
        except Exception as e:
            logger.error(f"Agent execution failed with timeout: {e}")
            return None
    
    def validate_agent_response(self, event) -> bool:
        """
        Validate that an agent response is properly formatted.
        
        Args:
            event: Agent response event
            
        Returns:
            True if response is valid, False otherwise
        """
        if not event:
            return False
        
        try:
            # Check if event has the expected structure
            if not hasattr(event, 'content'):
                logger.warning("Agent response missing content attribute")
                return False
            
            if not hasattr(event.content, 'parts'):
                logger.warning("Agent response content missing parts attribute")
                return False
            
            if not event.content.parts:
                logger.warning("Agent response has empty parts")
                return False
            
            # Check if we can access the text
            text = event.content.parts[0].text
            if not text or not text.strip():
                logger.warning("Agent response has empty text")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating agent response: {e}")
            return False
    
    def extract_response_text(self, event) -> Optional[str]:
        """
        Safely extract text from an agent response.
        
        Args:
            event: Agent response event
            
        Returns:
            Response text or None if extraction failed
        """
        if not self.validate_agent_response(event):
            return None
        
        try:
            return event.content.parts[0].text.strip()
        except Exception as e:
            logger.error(f"Failed to extract response text: {e}")
            return None
    
    def get_runner_stats(self, runner: Runner) -> dict:
        """
        Get statistics about a runner's performance.
        
        Args:
            runner: The runner instance
            
        Returns:
            Dictionary with runner statistics
        """
        try:
            return {
                "app_name": getattr(runner, 'app_name', 'unknown'),
                "agent_type": type(runner.agent).__name__ if hasattr(runner, 'agent') else 'unknown',
                "session_service_type": type(runner.session_service).__name__ if hasattr(runner, 'session_service') else 'unknown'
            }
        except Exception as e:
            logger.error(f"Failed to get runner stats: {e}")
            return {"error": str(e)}