"""
Configuration module for OptiMind system
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)


class OptimizationComplete(Exception):
    """Exception raised when optimization is completed by agent decision."""
    pass


@dataclass
class OptiMindConfig:
    """Configuration class for OptiMind optimization system."""
    
    # API Configuration
    model_name: str = "gemini-2.0-flash"
    
    # Application Configuration
    app_name: str = "opti_mind_tune"
    user_id: str = "user_001"
    
    # Optimization Parameters
    max_iterations: int = 2
    min_accuracy: float = 0.8
    target_accuracy: float = 0.95
    exploration_ratio: float = 0.3

    # Rate Limiting
    max_retries: int = 3
    rate_limit_delay: float = 1.0
    
    # Directory Configuration
    output_dir: Path = Path("output")
    logs_dir: Path = None
    csv_dir: Path = None
    
    # Session IDs for different agents
    session_ids: Dict[str, str] = None
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Set up directories
        if self.logs_dir is None:
            self.logs_dir = self.output_dir / "conversations"
        
        if self.csv_dir is None:
            self.csv_dir = self.output_dir / "csv_results"
        
        # Ensure directories exist
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up session IDs
        if self.session_ids is None:
            self.session_ids = {
                "recommender": "rec_session",
                "evaluator": "eval_session", 
                "decision": "dec_session"
            }
    
    def validate(self) -> bool:
        """
        Validate configuration parameters.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        
        if not (0 <= self.target_accuracy <= 1):
            raise ValueError("target_accuracy must be between 0 and 1")
        
        if not (0 <= self.min_accuracy <= 1):
            raise ValueError("min_accuracy must be between 0 and 1")
        
        if self.min_accuracy >= self.target_accuracy:
            raise ValueError("min_accuracy must be less than target_accuracy")
        
        if not (0 <= self.exploration_ratio <= 1):
            raise ValueError("exploration_ratio must be between 0 and 1")
        
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        
        if self.rate_limit_delay < 0:
            raise ValueError("rate_limit_delay must be non-negative")
        
        return True
    
    def to_dict(self) -> Dict:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            "model_name": self.model_name,
            "app_name": self.app_name,
            "user_id": self.user_id,
            "max_iterations": self.max_iterations,
            "target_accuracy": self.target_accuracy,
            "min_accuracy": self.min_accuracy,
            "exploration_ratio": self.exploration_ratio,
            "max_retries": self.max_retries,
            "rate_limit_delay": self.rate_limit_delay,
            "output_dir": str(self.output_dir),
            "logs_dir": str(self.logs_dir),
            "csv_dir": str(self.csv_dir),
            "session_ids": self.session_ids
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'OptiMindConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary with configuration parameters
            
        Returns:
            OptiMindConfig instance
        """
        # Convert string paths back to Path objects
        if "output_dir" in config_dict:
            config_dict["output_dir"] = Path(config_dict["output_dir"])
        if "logs_dir" in config_dict:
            config_dict["logs_dir"] = Path(config_dict["logs_dir"])
        if "csv_dir" in config_dict:
            config_dict["csv_dir"] = Path(config_dict["csv_dir"])
        
        return cls(**config_dict)
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"OptiMindConfig(model={self.model_name}, max_iter={self.max_iterations}, target_acc={self.target_accuracy})"
    
    def __repr__(self) -> str:
        """Detailed string representation of configuration."""
        return (f"OptiMindConfig("
                f"model_name='{self.model_name}', "
                f"max_iterations={self.max_iterations}, "
                f"target_accuracy={self.target_accuracy}, "
                f"min_accuracy={self.min_accuracy}, "
                f"app_name='{self.app_name}', "
                f"user_id='{self.user_id}'"
                f")")


# Default configuration instance
DEFAULT_CONFIG = OptiMindConfig()