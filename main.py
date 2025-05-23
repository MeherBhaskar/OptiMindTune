import json
import logging
import pandas as pd
import os
import re
import time
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
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from config import OptiMindConfig, OptimizationComplete
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

# Output directories
OUTPUT_DIR = Path("output")
LOGS_DIR = OUTPUT_DIR / "conversations"
CSV_DIR = OUTPUT_DIR / "csv_results"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

class ConfigEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        return json.JSONEncoder.default(self, obj)

def setup_directories():
    """Create necessary directories for output"""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

def save_conversation(data, timestamp: str):
    log_file = LOGS_DIR / f"conversation_{timestamp}.json"
    existing = {"interactions": []} if not log_file.exists() else json.load(open(log_file))
    existing["interactions"].append({**data, "timestamp": datetime.now().isoformat()})
    existing["last_updated"] = datetime.now().isoformat()
    json.dump(existing, open(log_file, 'w'), indent=2, cls=ConfigEncoder)

def run_agent(runner, message, session_id: str, config: OptiMindConfig):
    """Run agent with proper session ID and rate limiting"""
    rate_limiter = RateLimitHandler(
        calls_per_minute=10,
        max_retries=config.max_retries,
        initial_delay=config.rate_limit_delay
    )
    
    def _run_with_session():
        for event in runner.run(
            user_id=config.user_id,  # Use config's user_id
            session_id=session_id,
            new_message=message
        ):
            if event.is_final_response():
                return event
        return None

    try:
        return rate_limiter.with_retries(_run_with_session)
    except Exception as e:
        logger.error(f"Agent run failed after retries: {e}")
        return None

def get_recommendation_prompt(X, y, session):
    return f"""
Dataset Metadata:
n_samples: {X.shape[0]}
n_features: {X.shape[1]}
n_classes: {len(y.unique())}
class_balance: {y.value_counts(normalize=True).to_dict()}

Previous Results (if any):
{json.dumps(session.state.get("evaluation_history", []))}

Supported Models: RandomForestClassifier, LogisticRegression, SVC

Respond with a JSON object: {{"recommendations": [{{"model": "ModelName", "hyperparameters": "param1=value1, param2=value2", "reasoning": "Your reasoning"}}]}}
"""

def parse_recommendations(event):
    if not event:
        return []
    try:
        return json.loads(event.content.parts[0].text).get('recommendations', [])
    except:
        return []

def evaluate_model(runner, rec, iteration, timestamp, session_id, config):
    eval_input = f"Evaluate {rec['model']} with {rec['hyperparameters']}"
    eval_message = types.Content(role="user", parts=[types.Part(text=eval_input)])
    eval_event = run_agent(runner, eval_message, session_id, config)
    result = {"accuracy": 0.0, "success": False}
    if eval_event:
        try:
            result = json.loads(eval_event.content.parts[0].text)
            result["success"] = True
            logger.info(f"Model {rec['model']} evaluation: {result}")
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
    save_conversation({
        "iteration": iteration,
        "agent": "evaluator",
        "model": rec["model"],
        "input": eval_input,
        "output": result,
        "status": "success" if result.get("accuracy", 0) > 0 else "failed"
    }, timestamp)
    return result

def make_decision(runner, rec, eval_result, previous_results, iteration, timestamp, session_id, config):
    decision_input = f"""Decide whether to accept {rec['model']} with accuracy {eval_result['accuracy']:.4f}.
Previous results: {json.dumps(previous_results)}"""
    decision_message = types.Content(role="user", parts=[types.Part(text=decision_input)])
    decision_event = run_agent(runner, decision_message, session_id, config)
    if not decision_event:
        return None
    try:
        decision = json.loads(decision_event.content.parts[0].text)
        save_conversation({
            "iteration": iteration,
            "agent": "decision",
            "model": rec["model"],
            "input": decision_input,
            "output": decision,
            "status": "success"
        }, timestamp)
        return decision
    except Exception as e:
        logger.error(f"Decision failed: {e}")
        return None

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config = OptiMindConfig()

    # Set up directories from config
    config.logs_dir.mkdir(parents=True, exist_ok=True)

    # Load all datasets
    datasets = {
        'breast_cancer': load_breast_cancer(),
        'iris': load_iris(),
        'wine': load_wine(),

    }

    benchmark_results = {}
    csv_rows = []

    for dataset_name, data in datasets.items():
        logger.info(f"\nStarting optimization for {dataset_name} dataset")
        start_time = time.time()
        
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)

        # Create dataset-specific session IDs
        dataset_session_ids = {
            name: f"{session_id}_{dataset_name}"
            for name, session_id in config.session_ids.items()
        }

        # Initialize session service with config parameters
        session_service = InMemorySessionService()
        sessions = {
            name: session_service.create_session(
                app_name=config.app_name,
                user_id=config.user_id,
                session_id=session_id
            )
            for name, session_id in dataset_session_ids.items()
        }

        # Initialize agents with config
        recommender = RecommenderAgent(model=config.model_name)
        evaluator = EvaluationAgent(X, y, model=config.model_name)
        decision = DecisionAgent(model=config.model_name)

        runners = {
            name: Runner(
                app_name=config.app_name,
                agent=agent,
                session_service=session_service
            )
            for name, agent in [
                ("recommender", recommender),
                ("evaluator", evaluator),
                ("decision", decision)
            ]
        }

        best_model = None
        best_accuracy = 0.0

        try:
            for iteration in range(config.max_iterations):
                logger.info(f"Starting iteration {iteration + 1}/{config.max_iterations}")

                # Use the recommender session for previous_results and prompt
                previous_results = sessions["recommender"].state.get("evaluation_history", [])
                user_input = get_recommendation_prompt(X, y, sessions["recommender"])
                message = types.Content(role="user", parts=[types.Part(text=user_input)])
                
                # Run recommender with dataset-specific session ID
                event = run_agent(
                    runners["recommender"], 
                    message, 
                    dataset_session_ids["recommender"],
                    config
                )
                recommendations = parse_recommendations(event)

                if not recommendations:
                    logger.error("No valid recommendations received")
                    continue

                # Log recommender conversation
                save_conversation({
                    "iteration": iteration,
                    "agent": "recommender",
                    "dataset": dataset_name,
                    "input": user_input,
                    "output": recommendations,
                    "status": "success" if recommendations else "failed"
                }, timestamp)
                
                # Process each recommendation
                best_recommendation = None
                best_rec_accuracy = 0.0
                
                for rec in recommendations:
                    model_name = rec.get('model')
                    hyperparameters = rec.get('hyperparameters')
                    reasoning = rec.get('reasoning')
                    logger.info(f"Evaluating {model_name} with {hyperparameters} (Reason: {reasoning})")

                    # Run evaluation with dataset-specific session ID
                    eval_result = evaluate_model(
                        runners["evaluator"], 
                        rec, 
                        iteration,
                        timestamp,
                        dataset_session_ids["evaluator"],
                        config
                    )

                    if eval_result.get("accuracy", 0) == 0:
                        logger.warning(f"Skipping decision for failed evaluation of {model_name}")
                        continue

                    # Run decision with dataset-specific session ID
                    decision = make_decision(
                        runners["decision"],
                        rec,
                        eval_result,
                        previous_results,
                        iteration,
                        timestamp,
                        dataset_session_ids["decision"],
                        config
                    )

                    if decision and decision.get("accept"):
                        current_accuracy = decision.get("accuracy", 0.0)
                        if current_accuracy > best_rec_accuracy:
                            best_rec_accuracy = current_accuracy
                            best_recommendation = {
                                "model": model_name,
                                "hyperparameters": hyperparameters,
                                "accuracy": current_accuracy,
                                "accept": True,
                                "reasoning": decision.get("reasoning", "")
                            }

                # Update state with best recommendation from this iteration
                if best_recommendation:
                    if "evaluation_history" not in sessions["recommender"].state:
                        sessions["recommender"].state["evaluation_history"] = []
                    sessions["recommender"].state["evaluation_history"].append(best_recommendation)
                    
                    if best_rec_accuracy > best_accuracy:
                        best_accuracy = best_rec_accuracy
                        best_model = best_recommendation.copy()

                # Check optimization criteria
                if best_model:
                    if best_accuracy >= config.target_accuracy:
                        if iteration >= config.max_iterations * config.exploration_ratio:
                            logger.info(f"Found excellent model with accuracy {best_accuracy:.4f}, stopping search")
                            break
                    elif best_accuracy < config.min_accuracy:
                        logger.warning(f"Current best accuracy {best_accuracy:.4f} below minimum threshold")

        except OptimizationComplete:
            logger.info("Optimization completed successfully")
        finally:
            end_time = time.time()
            optimization_time = end_time - start_time
            
            # Store benchmark results for this dataset
            benchmark_results[dataset_name] = {
                "best_model": best_model["model"] if best_model else None,
                "best_hyperparameters": best_model["hyperparameters"] if best_model else None,
                "best_accuracy": best_accuracy,
                "total_iterations": iteration + 1,
                "optimization_time": optimization_time,
                "iterations_per_second": (iteration + 1) / optimization_time
            }

            # Add row for CSV
            csv_rows.append({
                "timestamp": timestamp,
                "dataset": dataset_name,
                "model": best_model["model"] if best_model else None,
                "best_accuracy": best_accuracy,
                "optimization_time": optimization_time,
                "iterations_per_second": (iteration + 1) / optimization_time,
                "total_iterations": iteration + 1,
                "hyperparameters": str(best_model["hyperparameters"] if best_model else None)
            })

            # Update final metadata with JSON-serializable config
            final_metadata = {
                "config": {k: str(v) if isinstance(v, Path) else v 
                          for k, v in vars(config).items()},
                "dataset": dataset_name,
                "total_iterations": iteration + 1,
                "best_model": best_model,
                "best_accuracy": best_accuracy,
                "optimization_time": optimization_time,
                "completed_at": datetime.now().isoformat()
            }
            
            # Save final metadata with custom encoder
            log_file = LOGS_DIR / f"conversation_{timestamp}.json"
            if log_file.exists():
                with open(log_file, "r") as f:
                    conversation_data = json.load(f)
                if "datasets" not in conversation_data:
                    conversation_data["datasets"] = {}
                conversation_data["datasets"][dataset_name] = final_metadata
                with open(log_file, "w") as f:
                    json.dump(conversation_data, f, indent=2, cls=ConfigEncoder)

    # Save results to CSV
    csv_file = CSV_DIR / f"optimind_benchmark_{timestamp}.csv"
    pd.DataFrame(csv_rows).to_csv(csv_file, index=False)
    logger.info(f"Results saved to CSV: {csv_file}")

    # Print final benchmark summary
    logger.info("\n=== Benchmark Summary ===")
    for dataset_name, results in benchmark_results.items():
        logger.info(f"\n{dataset_name.upper()} Dataset:")
        logger.info(f"Best Model: {results['best_model']}")
        logger.info(f"Best Hyperparameters: {results['best_hyperparameters']}")
        logger.info(f"Best Accuracy: {results['best_accuracy']:.4f}")
        logger.info(f"Total Iterations: {results['total_iterations']}")
        logger.info(f"Optimization Time: {results['optimization_time']:.2f} seconds")
        logger.info(f"Iterations/Second: {results['iterations_per_second']:.2f}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise