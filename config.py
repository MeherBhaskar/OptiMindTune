from dataclasses import dataclass, field
from typing import Dict
from pathlib import Path

@dataclass
class OptiMindConfig:
    # Optimization parameters
    max_iterations: int = 10
    min_accuracy: float = 0.8
    target_accuracy: float = 0.99
    exploration_ratio: float = 0.3

    # Session configuration
    app_name: str = "opti_mind_tune"
    user_id: str = "bhaskar_new"
    session_ids: Dict[str, str] = field(default_factory=lambda: {
        "recommender": "rec_session",
        "evaluator": "eval_session",
        "decision": "dec_session"
    })

    # Model configuration
    supported_models: list = field(default_factory=lambda: [
        "RandomForestClassifier",
        "LogisticRegression", 
        "SVC"
    ])

    # Output configuration
    output_dir: Path = Path("output")
    logs_dir: Path = Path("output/conversations")

    # Agent configuration
    model_name: str = "gemini-2.0-flash"
    rate_limit_delay: float = 1.0
    max_retries: int = 3

    def __post_init__(self):
        # Validate parameters
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if not (0 <= self.min_accuracy <= 1):
            raise ValueError("min_accuracy must be between 0 and 1")
        if not (0 <= self.target_accuracy <= 1):
            raise ValueError("target_accuracy must be between 0 and 1")
        if not (0 <= self.exploration_ratio <= 1):
            raise ValueError("exploration_ratio must be between 0 and 1")

class OptimizationComplete(Exception):
    """Raised when optimization meets target criteria"""
    pass
