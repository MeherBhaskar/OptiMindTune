from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import pandas as pd
from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import logging
from dotenv import load_dotenv
import os
import asyncio

# Load environment variables from the agent's .env file
load_dotenv()

logger = logging.getLogger(__name__)

class RecommenderAgent:
    def __init__(self):
        self.agent = Agent(
            name="recommender",
            model="gemini-2.0-flash",
            instruction="You are an expert AI agent for AutoML. Given the dataset metadata and previous evaluation results, recommend 1-2 scikit-learn classification models and specific hyperparameter configurations to try next. If no previous results exist, suggest initial models and hyperparameters. Provide a brief reasoning for each recommendation."
        )
        self.model_map = {
            "RandomForestClassifier": RandomForestClassifier,
            "LogisticRegression": LogisticRegression,
            "SVC": SVC
        }
        self.session_service = InMemorySessionService()
        self.runner = Runner(agent=self.agent, app_name="recommender_app", session_service=self.session_service)
        self.user_id = "user_recommender"
        self.session_id = "session_recommender"

    def get_dataset_metadata(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Extract metadata from the dataset."""
        return {
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "n_classes": len(y.unique()),
            "class_balance": y.value_counts(normalize=True).to_dict(),
        }

    def recommend(self, X: pd.DataFrame, y: pd.Series, previous_results: str = "") -> str:
        """Recommend models and hyperparameters using Gemini-2.0-flash."""
        metadata = self.get_dataset_metadata(X, y)
        dataset_info = "\n".join([f"{k}: {v}" for k, v in metadata.items()])
        user_input = f"""
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
        async def run_agent():
            self.session_service.create_session(app_name="recommender_app", user_id=self.user_id, session_id=self.session_id)
            message = types.Content(role="user", parts=[types.Part(text=user_input)])
            response_text = ""
            async for event in self.runner.run_async(user_id=self.user_id, session_id=self.session_id, new_message=message):
                if event.is_final_response():
                    response_text = event.content.parts[0].text
                    break
            return response_text

        response = asyncio.run(run_agent())
        logger.info(f"Recommender response: {response}")
        return response