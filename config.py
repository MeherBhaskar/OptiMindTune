from dataclasses import dataclass

@dataclass
class OptimizationConfig:
    max_iterations: int = 10
    min_accuracy: float = 0.8  # Minimum acceptable accuracy
    target_accuracy: float = 0.95  # Ideal accuracy to stop optimization
    exploration_ratio: float = 0.3  # Ratio of iterations to explore even after finding good model

class OptimizationComplete(Exception):
    """Raised when optimization meets target criteria"""
    pass
