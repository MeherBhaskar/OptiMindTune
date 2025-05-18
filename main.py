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

def save_conversation(conversation_history: list, timestamp: str, metadata: dict = None):
    """Save agent conversations with detailed information"""
    log_file = LOGS_DIR / f"conversation_{timestamp}.json"
    conversation_data = {
        "timestamp": timestamp,
        "metadata": metadata,
        "interactions": conversation_history
    }
    print('~~~~ Saving to log_file')
    print(conversation_data)
    with open(log_file, "w") as f:
        json.dump(conversation_data, f, indent=2)

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
            
            recommendations = []
            async for event in recommender_runner.run_async(user_id=user_id, session_id=session_id, new_message=message):
                if event.is_final_response():
                    try:
                        response_json = json.loads(event.content.parts[0].text)
                        recommendations = response_json.get('recommendations', [])
                        logger.info(f"Received recommendations: {recommendations}")
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse recommendations: {e}")
                        continue
                    break

            if not recommendations:
                logger.error("No valid recommendations received")
                continue

            # Log recommender conversation
            conversation_history.append({
                "iteration": iteration + 1,
                "agent": "recommender",
                "input": user_input,
                "output": recommendations
            })
            
            for rec in recommendations:
                model_name = rec.get('model')
                hyperparameters = rec.get('hyperparameters')
                reasoning = rec.get('reasoning')
                logger.info(f"Evaluating {model_name} with {hyperparameters} (Reason: {reasoning})")

                # Run evaluation
                eval_input = f"Evaluate {model_name} with {hyperparameters} using evaluate_model tool. Return JSON: {{'accuracy': float}}"
                eval_message = types.Content(role="user", parts=[types.Part(text=eval_input)])
                async for eval_event in evaluation_runner.run_async(user_id=user_id, session_id=session_id, new_message=eval_message):
                    if eval_event.is_final_response():
                        try:
                            eval_response = json.loads(eval_event.content.parts[0].text)
                            accuracy = eval_response["accuracy"]
                            # Get the trained pipeline from the agent
                            model_object = evaluation_agent.get_current_pipeline()
                        except (json.JSONDecodeError, KeyError) as e:
                            logger.error(f"Failed to parse evaluation response: {e}")
                            accuracy = 0.0
                            model_object = None
                        break
                else:
                    logger.error("No evaluation response")
                    continue

                # Log evaluation conversation
                conversation_history.append({
                    "iteration": iteration + 1,
                    "agent": "evaluator",
                    "model": model_name,
                    "input": eval_input,
                    "output": {"accuracy": accuracy}
                })
                
                # Run decision
                decision_input = f"Decide whether to accept {model_name} with accuracy {accuracy}. Previous results: {json.dumps(previous_results)}"
                decision_message = types.Content(role="user", parts=[types.Part(text=decision_input)])
                async for decision_event in decision_runner.run_async(user_id=user_id, session_id=session_id, new_message=decision_message):
                    if decision_event.is_final_response():
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
                            
                            # Log decision conversation
                            conversation_history.append({
                                "iteration": iteration + 1,
                                "agent": "decision",
                                "input": decision_input,
                                "output": decision
                            })

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
        # Save final results
        final_metadata = {
            "config": vars(config),
            "total_iterations": iteration + 1,
            "best_model": best_model,
            "best_accuracy": best_accuracy
        }
        save_conversation(conversation_history, timestamp, final_metadata)
        logger.info(f"Optimization completed after {iteration + 1} iterations")
        logger.info(f"Best model found: {best_model}")

if __name__ == "__main__":
    asyncio.run(main())