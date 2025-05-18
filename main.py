import asyncio
import json
import logging
import pandas as pd
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
from agents.recommender import RecommenderAgent
from agents.evaluation import EvaluationAgent
from agents.decision import DecisionAgent
from sklearn.datasets import load_iris
from config import OptimizationConfig, OptimizationComplete
from typing import List, Dict, Any
from utils.rate_limiter import RateLimitHandler

# Load environment variables
load_dotenv()

# Configure Google AI
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set")
genai.configure(api_key=api_key)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Only keep conversation logging
OUTPUT_DIR = Path("output")
LOGS_DIR = OUTPUT_DIR / "conversations"

def setup_directories():
    """Create necessary directories for output"""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

async def save_conversation_increment(conversation_data: Dict[str, Any], timestamp: str):
    """Save agent conversation incrementally"""
    log_file = LOGS_DIR / f"conversation_{timestamp}.json"
    
    # Load existing conversations if file exists
    existing_data = {"interactions": []}
    if log_file.exists():
        with open(log_file, "r") as f:
            existing_data = json.load(f)
    
    # Append new conversation
    existing_data["interactions"].append(conversation_data)
    existing_data["last_updated"] = datetime.now().isoformat()
    
    # Save updated conversations
    with open(log_file, "w") as f:
        json.dump(existing_data, f, indent=2)

async def main():
    # Setup output directory
    setup_directories()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize configuration
    config = OptimizationConfig(
        max_iterations=5,  # Adjust default values here
        min_accuracy=0.85,
        target_accuracy=0.95,
        exploration_ratio=0.3
    )

    # Initialize conversation history
    conversation_history = []

    # Load dataset (replace with your dataset)
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    # Set up session
    session_service = InMemorySessionService()
    session_id = "session123"
    user_id = "user123"
    session = session_service.create_session(app_name="opti_mind_tune", user_id=user_id, session_id=session_id)

    # Initialize agents
    recommender_agent = RecommenderAgent()
    evaluation_agent = EvaluationAgent(X, y)
    decision_agent = DecisionAgent()

    # Initialize runners
    recommender_runner = Runner(agent=recommender_agent, app_name="opti_mind_tune", session_service=session_service)
    evaluation_runner = Runner(agent=evaluation_agent, app_name="opti_mind_tune", session_service=session_service)
    decision_runner = Runner(agent=decision_agent, app_name="opti_mind_tune", session_service=session_service)

    # Add rate limiting to API calls
    rate_limit_handler = RateLimitHandler(base_delay=2.0, max_retries=3)

    @rate_limit_handler
    async def run_agent(runner, user_id: str, session_id: str, message: types.Content):
        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=message):
            if event.is_final_response():
                return event
        return None

    max_iterations = 2
    best_model = None
    best_accuracy = 0.0

    try:
        for iteration in range(config.max_iterations):
            logger.info(f"Starting iteration {iteration + 1}/{config.max_iterations}")

            # Run recommender
            previous_results = session.state.get("evaluation_history", [])
            user_input = f"""
Dataset Metadata:
n_samples: {X.shape[0]}
n_features: {X.shape[1]}
n_classes: {len(y.unique())}
class_balance: {y.value_counts(normalize=True).to_dict()}

Previous Results (if any):
{json.dumps(previous_results)}

Supported Models: RandomForestClassifier, LogisticRegression, SVC

Respond with a JSON object: {{"recommendations": [{{"model": "ModelName", "hyperparameters": "param1=value1, param2=value2", "reasoning": "Your reasoning"}}]}}
"""
            message = types.Content(role="user", parts=[types.Part(text=user_input)])
            
            # Run recommender with rate limiting
            event = await run_agent(recommender_runner, user_id, session_id, message)
            if event:
                try:
                    response_json = json.loads(event.content.parts[0].text)
                    recommendations = response_json.get('recommendations', [])
                    logger.info(f"Received recommendations: {recommendations}")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse recommendations: {e}")
                    continue

            if not recommendations:
                logger.error("No valid recommendations received")
                continue

            # Log recommender conversation immediately
            await save_conversation_increment({
                "timestamp": datetime.now().isoformat(),
                "iteration": iteration + 1,
                "agent": "recommender",
                "input": user_input,
                "output": recommendations,
                "status": "success" if recommendations else "failed"
            }, timestamp)
            
            for rec in recommendations:
                model_name = rec.get('model')
                hyperparameters = rec.get('hyperparameters')
                reasoning = rec.get('reasoning')
                logger.info(f"Evaluating {model_name} with {hyperparameters} (Reason: {reasoning})")

                # Run evaluation
                eval_input = f"Evaluate {model_name} with {hyperparameters} using evaluate_model tool. Return JSON: {{'accuracy': float}}"
                eval_message = types.Content(role="user", parts=[types.Part(text=eval_input)])
                
                # Run evaluation with rate limiting
                eval_event = await run_agent(evaluation_runner, user_id, session_id, eval_message)
                eval_accuracy = 0.0
                eval_response = {"accuracy": 0.0, "success": False, "error": "No response"}

                if eval_event:
                    try:
                        eval_response = json.loads(eval_event.content.parts[0].text)
                        if eval_response.get("success", False):
                            eval_accuracy = float(eval_response.get("accuracy", 0.0))
                            logger.info(f"Evaluation success - Accuracy: {eval_accuracy:.4f}")
                        else:
                            error_msg = eval_response.get("error", "Unknown error")
                            logger.error(f"Evaluation failed: {error_msg}")
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        logger.error(f"Failed to parse evaluation response: {e}")
                        eval_response = {
                            "accuracy": 0.0,
                            "success": False,
                            "error": str(e)
                        }

                # Log evaluation conversation
                await save_conversation_increment({
                    "timestamp": datetime.now().isoformat(),
                    "iteration": iteration + 1,
                    "agent": "evaluator",
                    "model": model_name,
                    "input": eval_input,
                    "output": eval_response,
                    "status": "success" if eval_accuracy > 0 else "failed"
                }, timestamp)
                
                # Run decision
                decision_input = f"Decide whether to accept {model_name} with accuracy {eval_accuracy}. Previous results: {json.dumps(previous_results)}"
                decision_message = types.Content(role="user", parts=[types.Part(text=decision_input)])
                
                # Run decision with rate limiting
                decision_event = await run_agent(decision_runner, user_id, session_id, decision_message)
                if decision_event:
                    try:
                        decision = json.loads(decision_event.content.parts[0].text)
                        # Store in session state
                        if "evaluation_history" not in session.state:
                            session.state["evaluation_history"] = []
                        
                        # Add to evaluation history
                        current_result = {
                            "model": model_name,
                            "hyperparameters": hyperparameters,
                            "accuracy": decision["accuracy"],
                            "accept": decision["accept"],
                            "reasoning": decision["reasoning"]
                        }
                        session.state["evaluation_history"].append(current_result)
                        
                        # Log decision conversation immediately
                        await save_conversation_increment({
                            "timestamp": datetime.now().isoformat(),
                            "iteration": iteration + 1,
                            "agent": "decision",
                            "model": model_name,
                            "input": decision_input,
                            "output": decision,
                            "status": "success"
                        }, timestamp)

                        logger.info(f"Result: Accuracy={decision['accuracy']:.4f}, Accept={decision['accept']}, Reasoning={decision['reasoning']}")
                        
                        # Update best model if better accuracy found
                        if decision["accuracy"] > best_accuracy:
                            best_accuracy = decision["accuracy"]
                            best_model = current_result.copy()
                            
                        # Only break from decision loop, not entire iteration
                        break
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.error(f"Failed to parse decision response: {e}")
                    break

            # Check optimization criteria
            if best_model:
                if best_accuracy >= config.target_accuracy:
                    # Allow some exploration based on exploration_ratio
                    if iteration >= config.max_iterations * config.exploration_ratio:
                        logger.info(f"Found excellent model with accuracy {best_accuracy:.4f}, stopping search")
                        raise OptimizationComplete
                elif best_accuracy < config.min_accuracy:
                    logger.warning(f"Current best accuracy {best_accuracy:.4f} below minimum threshold")

    except OptimizationComplete:
        logger.info("Optimization completed successfully")
    finally:
        # Update final metadata
        final_metadata = {
            "config": vars(config),
            "total_iterations": iteration + 1,
            "best_model": best_model,
            "best_accuracy": best_accuracy,
            "completed_at": datetime.now().isoformat()
        }
        
        # Save final metadata
        log_file = LOGS_DIR / f"conversation_{timestamp}.json"
        if log_file.exists():
            with open(log_file, "r") as f:
                conversation_data = json.load(f)
            conversation_data["metadata"] = final_metadata
            with open(log_file, "w") as f:
                json.dump(conversation_data, f, indent=2)

        logger.info(f"Optimization completed after {iteration + 1} iterations")
        logger.info(f"Best model found: {best_model}")

if __name__ == "__main__":
    asyncio.run(main())