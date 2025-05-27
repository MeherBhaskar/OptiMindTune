"""
Session Manager: Handles agent session creation and management
"""

import logging
from typing import Dict
from google.adk.sessions import InMemorySessionService
from config import OptiMindConfig

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages agent sessions for different datasets and agents."""
    
    def __init__(self, config: OptiMindConfig):
        self.config = config
        self.session_service = InMemorySessionService()
    
    def create_dataset_sessions(self, dataset_name: str) -> Dict[str, any]:
        """
        Create sessions for all agents for a specific dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary mapping agent names to their sessions
        """
        sessions = {}
        
        for agent_name in ['recommender', 'evaluator', 'decision']:
            session_id = f"{self.config.session_ids[agent_name]}_{dataset_name}"
            
            try:
                session = self.session_service.create_session(
                    app_name=self.config.app_name,
                    user_id=self.config.user_id,
                    session_id=session_id
                )
                sessions[agent_name] = session
                logger.debug(f"Created session for {agent_name} on {dataset_name}: {session_id}")
                
            except Exception as e:
                logger.error(f"Failed to create session for {agent_name} on {dataset_name}: {e}")
                raise
        
        return sessions
    
    def cleanup_sessions(self, sessions: Dict[str, any]):
        """
        Cleanup sessions when optimization is complete.
        
        Args:
            sessions: Dictionary of sessions to cleanup
        """
        for agent_name, session in sessions.items():
            try:
                # Sessions are automatically cleaned up by InMemorySessionService
                # but we can clear state if needed
                if hasattr(session, 'state'):
                    session.state.clear()
                logger.debug(f"Cleaned up session for {agent_name}")
            except Exception as e:
                logger.warning(f"Failed to cleanup session for {agent_name}: {e}")
    
    def get_session_info(self, sessions: Dict[str, any]) -> Dict[str, Dict]:
        """
        Get information about current sessions.
        
        Args:
            sessions: Dictionary of active sessions
            
        Returns:
            Dictionary with session information
        """
        session_info = {}
        
        for agent_name, session in sessions.items():
            try:
                info = {
                    "session_id": getattr(session, 'session_id', 'unknown'),
                    "app_name": getattr(session, 'app_name', 'unknown'),
                    "user_id": getattr(session, 'user_id', 'unknown'),
                    "state_keys": list(session.state.keys()) if hasattr(session, 'state') else [],
                    "state_size": len(session.state) if hasattr(session, 'state') else 0
                }
                session_info[agent_name] = info
            except Exception as e:
                logger.warning(f"Failed to get info for session {agent_name}: {e}")
                session_info[agent_name] = {"error": str(e)}
        
        return session_info