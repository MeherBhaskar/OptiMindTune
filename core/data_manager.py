"""
Data Manager: Handles dataset loading and preprocessing
"""

import logging
import pandas as pd
from typing import Dict, Tuple
from sklearn.datasets import load_iris, load_wine, load_breast_cancer

logger = logging.getLogger(__name__)


class DatasetManager:
    """Manages dataset loading and preprocessing."""
    
    def __init__(self):
        self.available_datasets = {
            'iris': load_iris,
            'wine': load_wine,
            'breast_cancer': load_breast_cancer
        }
    
    def load_datasets(self, dataset_name) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """
        Load available datasets.
        
        Returns:
            Dictionary mapping dataset names to (X, y) tuples
        """
        datasets = {}
        
        for name, loader_func in self.available_datasets.items():
            if dataset_name == name:
                try:
                    X, y = self.load_dataset(name, loader_func)
                    datasets[name] = (X, y)
                    logger.info(f"Loaded {name} dataset: {X.shape[0]} samples, {X.shape[1]} features")
                except Exception as e:
                    logger.error(f"Failed to load {name} dataset: {e}")
        
        return datasets

    def load_dataset(self, name: str, loader_func) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load a single dataset using the provided loader function.
        
        Args:
            name: Name of the dataset
            loader_func: Function to load the dataset
            
        Returns:
            Tuple of (features, target) as pandas DataFrame and Series
        """
        try:
            data = loader_func()
            
            # Convert to pandas for consistent interface
            X = pd.DataFrame(data.data, columns=data.feature_names)
            y = pd.Series(data.target, name='target')
            
            # Basic validation
            self._validate_dataset(name, X, y)
            return X,y
            
        except Exception as e:
            logger.error(f"Error loading {name} dataset: {e}")
            raise
    
    def _validate_dataset(self, name: str, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Validate dataset integrity.
        
        Args:
            name: Dataset name
            X: Feature matrix
            y: Target vector
        """
        if X.empty:
            raise ValueError(f"Dataset {name} has empty feature matrix")
        
        if y.empty:
            raise ValueError(f"Dataset {name} has empty target vector")
        
        if len(X) != len(y):
            raise ValueError(f"Dataset {name} has mismatched X and y lengths: {len(X)} vs {len(y)}")
        
        if X.isnull().any().any():
            logger.warning(f"Dataset {name} contains missing values in features")
        
        if y.isnull().any():
            logger.warning(f"Dataset {name} contains missing values in target")
        
        # Log dataset statistics
        logger.debug(f"Dataset {name} statistics:")
        logger.debug(f"  - Samples: {len(X)}")
        logger.debug(f"  - Features: {X.shape[1]}")
        logger.debug(f"  - Classes: {len(y.unique())}")
        logger.debug(f"  - Class distribution: {y.value_counts().to_dict()}")
    
    def get_dataset_info(self, name: str, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Get comprehensive information about a dataset.
        
        Args:
            name: Dataset name
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dictionary with dataset information
        """
        return {
            "name": name,
            "n_samples": len(X),
            "n_features": X.shape[1],
            "n_classes": len(y.unique()),
            "class_distribution": y.value_counts(normalize=True).to_dict(),
            "feature_names": list(X.columns),
            "target_classes": sorted(y.unique()),
            "has_missing_features": X.isnull().any().any(),
            "has_missing_targets": y.isnull().any()
        }