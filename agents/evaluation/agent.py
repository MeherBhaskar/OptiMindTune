from google.adk.agents import Agent
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import logging
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, PrivateAttr
from ..base_agent import BaseLoggingAgent
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvaluateModelTool:
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y
        self.model_map = {
            "RandomForestClassifier": RandomForestClassifier,
            "LogisticRegression": LogisticRegression,
            "SVC": SVC
        }
        self.__name__ = "evaluate_model"
        self.__doc__ = "Evaluate a scikit-learn model with given hyperparameters using cross-validation."
        self.current_pipeline: Optional[Pipeline] = None
        
    def get_current_pipeline(self) -> Optional[Pipeline]:
        """Return the currently fitted pipeline"""
        return self.current_pipeline

    def parse_hyperparameters(self, hyperparameters: str) -> dict:
        """Parse hyperparameters string into dictionary with proper types"""
        params = {}
        try:
            for param in hyperparameters.split(", "):
                key, value = param.split("=")
                key = key.strip()
                value = value.strip("'\"")
                
                # Convert to appropriate type
                if value.lower() == 'true':
                    params[key] = True
                elif value.lower() == 'false':
                    params[key] = False
                elif value.isdigit():
                    params[key] = int(value)
                elif value.replace(".", "").isdigit():
                    params[key] = float(value)
                else:
                    params[key] = value
            return params
        except Exception as e:
            logger.error(f"Failed to parse hyperparameters: {hyperparameters} - {str(e)}")
            raise ValueError(f"Invalid hyperparameter format: {str(e)}")

    def __call__(self, model_name: str, hyperparameters: str) -> Dict[str, Any]:
        """Evaluate model and return results as dictionary"""
        try:
            if model_name not in self.model_map:
                raise ValueError(f"Unsupported model: {model_name}")
            
            hyperparams = self.parse_hyperparameters(hyperparameters)
            model = self.model_map[model_name](**hyperparams)
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", model)
            ])
            
            scores = cross_val_score(pipeline, self.X, self.y, cv=5, scoring="accuracy")
            mean_accuracy = float(scores.mean())
            
            pipeline.fit(self.X, self.y)
            self.current_pipeline = pipeline
            
            logger.info(f"Model {model_name} achieved accuracy: {mean_accuracy:.4f}")
            return {"accuracy": mean_accuracy}
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            return {"accuracy": 0.0, "error": str(e)}

class EvaluationAgent(Agent):
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self._evaluate_tool = EvaluateModelTool(X, y)
        super().__init__(
            name="evaluation",
            model="gemini-2.0-flash",
            instruction="""You are an evaluation agent. Run the evaluate_model tool and return its results EXACTLY as received.
            Do not modify the accuracy value. Return in format: {"accuracy": <value>}""",
            tools=[self._evaluate_tool],
            disallow_transfer_to_parent=True,
            disallow_transfer_to_peers=True
        )

    def get_current_pipeline(self) -> Optional[Pipeline]:
        return self._evaluate_tool.get_current_pipeline()