from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pandas as pd
from typing import Dict, Any
import logging
from dotenv import load_dotenv
import os
import asyncio

# Load environment variables from the agent's .env file
load_dotenv()

logger = logging.getLogger(__name__)

class EvaluateModelTool(AgentTool):
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y
        self.model_map = {
            "RandomForestClassifier": RandomForestClassifier,
            "LogisticRegression": LogisticRegression,
            "SVC": SVC
        }

    def __call__(self, model_name: str, hyperparameters: Dict[str, Any]) -> float:
        """Evaluate a model with given hyperparameters using cross-validation."""
        if model_name not in self.model_map:
            raise ValueError(f"Unsupported model: {model_name}")
        model_class = self.model_map[model_name]
        # Adjust max_iter for LogisticRegression with saga and l1
        if model_name == "LogisticRegression" and hyperparameters.get("solver") == "saga" and hyperparameters.get("penalty") == "l1":
            hyperparameters["max_iter"] = hyperparameters.get("max_iter", 1000)
        model = model_class(**hyperparameters)
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", model)
        ])
        try:
            scores = cross_val_score(pipeline, self.X, self.y, cv=5, scoring="accuracy")
            mean_accuracy = scores.mean()
            logger.info(f"Evaluated {model_name} with {hyperparameters}, Accuracy: {mean_accuracy:.4f}")
            return mean_accuracy
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {str(e)}")
            return 0.0

class EvaluatorAgent:
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.tool = EvaluateModelTool(X, y)
        self.agent = Agent(
            name="evaluator",
            model="gemini-2.0-flash",
            instruction="You are an expert AutoML evaluator. When given a model name and hyperparameters, use the 'evaluate_model' tool to evaluate it on the dataset. Based on the accuracy, dataset characteristics, and previous results, suggest whether to accept this model or recommend new ones. Return a JSON object with 'accuracy' (float), 'accept' (boolean), and 'reasoning' (string).",
            tools=[self.tool]
        )
        self.session_service = InMemorySessionService()
        self.runner = Runner(agent=self.agent, app_name="evaluator_app", session_service=self.session_service)
        self.user_id = "user_evaluator"
        self.session_id = "session_evaluator"

    def evaluate_and_suggest(self, model_name: str, hyperparameters: Dict[str, Any], previous_results: str) -> str:
        """Evaluate a model and suggest next steps using Gemini-2.0-flash."""
        hyperparams_str = ", ".join([f"{k}={v}" for k, v in hyperparameters.items()])
        user_input = f"""
Evaluate {model_name} with hyperparameters {hyperparams_str} using the 'evaluate_model' tool.
Previous Results:
{previous_results}
Based on the accuracy, dataset characteristics, and previous results, indicate whether to accept this model or recommend new ones.
Return a JSON object with keys 'accuracy' (float), 'accept' (boolean), and 'reasoning' (string).
"""
        async def run_agent():
            self.session_service.create_session(app_name="evaluator_app", user_id=self.user_id, session_id=self.session_id)
            message = types.Content(role="user", parts=[types.Part(text=user_input)])
            response_text = ""
            async for event in self.runner.run_async(user_id=self.user_id, session_id=self.session_id, new_message=message):
                if event.is_final_response():
                    response_text = event.content.parts[0].text
                    break
            return response_text

        response = asyncio.run(run_agent())
        logger.info(f"Evaluator response: {response}")
        return response