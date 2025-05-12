from typing import Dict, Any, List, Tuple
import pandas as pd
import json
import logging
import re
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from dotenv import load_dotenv
import os
import optuna

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

class RecommenderAgent:
    def __init__(self):
        """Initialize the LangChain agent with Mistral-7B."""
        self.llm = HuggingFaceEndpoint(
            repo_id="google/flan-t5-small",
            temperature=0.7,
            model_kwargs={"max_length": 512},
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )
        self.prompt_template = PromptTemplate(
            input_variables=["dataset_info", "previous_results"],
            template="""
You are an expert AI agent for AutoML. Given the dataset metadata and previous evaluation results, recommend 1-2 scikit-learn classification models and specific hyperparameter configurations to try next. If no previous results exist, suggest initial models and hyperparameters. Provide a brief reasoning for each recommendation.

Dataset Metadata:
{dataset_info}

Previous Results (if any):
{previous_results}

Supported Models: RandomForestClassifier, LogisticRegression, SVC

Output a JSON array of objects, strictly following this format:
[
  {{
    "model": "[Model Name]",
    "hyperparameters": "example: n_estimators=100, max_depth=5",
    "reasoning": "[Reasoning text]"
  }}
]
Do not include any text outside the JSON array.
"""
        )
        self.chain = (RunnablePassthrough() | self.prompt_template | self.llm)
        self.model_map = {
            "RandomForestClassifier": RandomForestClassifier,
            "LogisticRegression": LogisticRegression,
            "SVC": SVC
        }

    def get_dataset_metadata(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Extract metadata from the dataset."""
        return {
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "n_classes": len(y.unique()),
            "class_balance": y.value_counts(normalize=True).to_dict(),
        }

    def recommend(self, X: pd.DataFrame, y: pd.Series, previous_results: str = "") -> List[Dict[str, Any]]:
        """Recommend models and hyperparameters based on dataset and past results."""
        metadata = self.get_dataset_metadata(X, y)
        dataset_info = "\n".join([f"{k}: {v}" for k, v in metadata.items()])
        logger.info("Invoking LLM with dataset_info and previous_results")
        response = self.chain.invoke({"dataset_info": dataset_info, "previous_results": previous_results})
        response_text = response  # Response is a string directly

        logger.info(f"LLM Response: {response_text}")
        recommendations = []

        try:
            # Parse JSON response
            recommendations = json.loads(response_text)
            logger.info(f"Parsed Recommendations: {recommendations}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}, Raw response: {response_text}")
            return []

        # Convert hyperparameters string to dictionary for supported models
        parsed_recommendations = []
        for rec in recommendations:
            if rec.get("model") not in self.model_map:
                continue
            hyperparams_str = rec.get("hyperparameters", "")
            hyperparams_dict = {}
            try:
                # Parse string like "n_estimators=100, max_depth=5" or "C=1.0, kernel='rbf'"
                for pair in hyperparams_str.split(","):
                    if "=" in pair:
                        key_value = pair.strip().split("=", 1)
                        if len(key_value) != 2:
                            continue
                        key, value = key_value
                        key = key.strip()
                        # Remove quotes from value if present
                        value = re.sub(r'^["\'](.*)["\']$', r'\1', value.strip())
                        # Convert value to int/float if possible
                        try:
                            value = int(value)
                        except ValueError:
                            try:
                                value = float(value)
                            except ValueError:
                                pass  # Keep as string if not numeric
                        hyperparams_dict[key] = value
            except Exception as e:
                logger.error(f"Failed to parse hyperparameters: {e}, Hyperparameters: {hyperparams_str}")
                continue
            rec["hyperparameters"] = hyperparams_dict
            parsed_recommendations.append(rec)

        return parsed_recommendations

class EvaluatorAgent:
    def __init__(self):
        """Initialize the evaluator with a scikit-learn pipeline."""
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", None)  # Placeholder, set during evaluation
        ])

    def evaluate(self, X: pd.DataFrame, y: pd.Series, model_name: str, hyperparameters: Dict[str, Any], model_map: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Evaluate a model with given hyperparameters using cross-validation."""
        model_class = model_map.get(model_name)
        if not model_class:
            raise ValueError(f"Unsupported model: {model_name}")

        # Adjust max_iter for LogisticRegression with saga and l1
        if model_name == "LogisticRegression" and hyperparameters.get("solver") == "saga" and hyperparameters.get("penalty") == "l1":
            hyperparameters["max_iter"] = hyperparameters.get("max_iter", 1000)

        # Set model with hyperparameters
        model = model_class(**hyperparameters)
        self.pipeline.set_params(classifier=model)

        try:
            # Perform 5-fold cross-validation
            scores = cross_val_score(self.pipeline, X, y, cv=5, scoring="accuracy", error_score="raise")
            mean_accuracy = scores.mean()
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {str(e)}")
            return 0.0, {"model": model_name, "hyperparameters": hyperparameters, "accuracy": 0.0}

        # Log evaluation result
        logger.info(f"Evaluated {model_name} with {hyperparameters}, Accuracy: {mean_accuracy:.4f}")
        return mean_accuracy, {
            "model": model_name,
            "hyperparameters": hyperparameters,
            "accuracy": mean_accuracy
        }