from typing import List, Dict, Any
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler # Added
from sklearn.pipeline import Pipeline # Added
from sklearn.ensemble import RandomForestClassifier # Added
from sklearn.linear_model import LogisticRegression # Added
from sklearn.svm import SVC # Added
from agents.recommender.agent import RecommenderAgent
from agents.evaluator.agent import EvaluatorAgent
import json
import logging
from dotenv import load_dotenv
import os
import joblib # Added
import time # Added

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model map for re-instantiating models before saving
MODEL_CLASS_MAP = {
    "RandomForestClassifier": RandomForestClassifier,
    "LogisticRegression": LogisticRegression,
    "SVC": SVC
}

def parse_llm_response(response_str: str) -> Any:
    """Clean and parse LLM response, removing code block markers if present."""
    if response_str.startswith("```json") and response_str.endswith("```"):
        json_str = response_str[len("```json"):-len("```")].strip()
    else:
        json_str = response_str.strip()
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}, Raw response: {response_str}")
        return None

def run_optimization(X: pd.DataFrame, y: pd.Series, max_iterations: int = 10, accuracy_threshold: float = 0.99) -> List[Dict[str, Any]]:
    """Run the multi-agent optimization loop and save artifacts."""
    recommender = RecommenderAgent()
    evaluator = EvaluatorAgent(X, y)
    results = []
    previous_results = ""

    # Create artifact directories
    artifacts_dir = "artifacts"
    models_dir = os.path.join(artifacts_dir, "models")
    results_dir = os.path.join(artifacts_dir, "results")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    for iteration in range(max_iterations):
        logger.info(f"Iteration {iteration + 1}/{max_iterations}")

        # Get recommendations
        recommendations_str = recommender.recommend(X, y, previous_results)
        recommendations = parse_llm_response(recommendations_str)
        if recommendations is None:
            recommendations = []

        # Parse recommendations
        parsed_recommendations = []
        for rec in recommendations:
            if rec.get("model") not in recommender.model_map: # Checking against recommender's map for supported models
                logger.warning(f"Unsupported model recommended: {rec.get('model')}")
                continue
            hyperparams_str = rec.get("hyperparameters", "")
            hyperparams_dict = {}
            try:
                for pair in hyperparams_str.split(","):
                    if "=" in pair:
                        key_value = pair.strip().split("=", 1)
                        if len(key_value) != 2:
                            continue
                        key, value = key_value
                        key = key.strip()
                        value = value.strip().strip("'\"")
                        try:
                            value = int(value)
                        except ValueError:
                            try:
                                value = float(value)
                            except ValueError:
                                pass # Keep as string if not int or float
                        hyperparams_dict[key] = value
            except Exception as e:
                logger.error(f"Failed to parse hyperparameters: {e}, Hyperparameters: {hyperparams_str}")
                continue
            rec["hyperparameters"] = hyperparams_dict
            parsed_recommendations.append(rec)

        if not parsed_recommendations:
            logger.info("No valid recommendations received in this iteration.")
            # Continue to next iteration or break if needed, here we continue
            # If you want to stop if no recommendations are ever made, add a counter or flag
            if not results: # If no results yet and no new recommendations, then maybe stop
                 logger.warning("No recommendations received and no prior results. Stopping.")
                 break
            # If there are previous results, the recommender might learn, so continue
            # previous_results remains the same for the next iteration if no new iteration_results are added

        iteration_summary_for_recommender = []
        best_iteration_accuracy = -1.0
        stop_due_to_threshold = False

        for rec_idx, rec in enumerate(parsed_recommendations):
            model_name = rec["model"]
            hyperparameters = rec["hyperparameters"]
            logger.info(f"Evaluating {model_name} with {hyperparameters}")

            eval_response = evaluator.evaluate_and_suggest(model_name, hyperparameters, previous_results)
            eval_result = parse_llm_response(eval_response)
            if eval_result is None:
                logger.warning(f"Failed to parse evaluation for {model_name}. Skipping.")
                continue

            accuracy = eval_result.get("accuracy", 0.0)
            accept = eval_result.get("accept", False)
            reasoning = eval_result.get("reasoning", "No reasoning provided")
            
            current_result = {
                "iteration": iteration + 1,
                "recommendation_index": rec_idx +1,
                "model": model_name,
                "hyperparameters": hyperparameters, # These are the original recommended hyperparams
                "accuracy": accuracy,
                "accept": accept,
                "reasoning": reasoning,
                "saved_model_path": None # Initialize with None
            }

            if accept:
                logger.info(f"Model {model_name} with {hyperparameters} was accepted. Accuracy: {accuracy:.4f}.")
                # Re-train and save the accepted model
                try:
                    model_class = MODEL_CLASS_MAP.get(model_name)
                    if not model_class:
                        logger.error(f"Model class for {model_name} not found in MODEL_CLASS_MAP. Cannot save.")
                    else:
                        # Apply specific hyperparameter adjustments if necessary (e.g., for LogisticRegression)
                        # This ensures consistency with EvaluateModelTool's internal adjustments if any were critical for training
                        # For now, we directly use the `hyperparameters` dict. If EvaluateModelTool makes critical internal changes
                        # to hyperparams before fitting, those changes should ideally be returned or handled consistently.
                        # The example EvaluateModelTool does modify 'max_iter' for LogisticRegression under specific conditions.
                        # We should replicate that here for consistency IF these hyperparameters are used for saving.
                        
                        current_train_hyperparams = hyperparameters.copy() # Use a copy for training
                        if model_name == "LogisticRegression" and \
                           current_train_hyperparams.get("solver") == "saga" and \
                           current_train_hyperparams.get("penalty") == "l1":
                            current_train_hyperparams["max_iter"] = current_train_hyperparams.get("max_iter", 1000) # Default from tool

                        model_instance = model_class(**current_train_hyperparams)
                        pipeline_to_save = Pipeline([
                            ("scaler", StandardScaler()),
                            ("classifier", model_instance)
                        ])
                        pipeline_to_save.fit(X, y)
                        
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        model_filename = f"{model_name}_iter{iteration+1}_rec{rec_idx+1}_{timestamp}.joblib"
                        model_filepath = os.path.join(models_dir, model_filename)
                        joblib.dump(pipeline_to_save, model_filepath)
                        current_result["saved_model_path"] = model_filepath
                        logger.info(f"Saved accepted model to {model_filepath}")
                except Exception as e:
                    logger.error(f"Error saving model {model_name} with {hyperparameters}: {e}")
            
            results.append(current_result)
            iteration_summary_for_recommender.append(
                f"Model: {model_name}, Hyperparameters: {hyperparameters}, Accuracy: {accuracy:.4f}, Accept: {accept}, Reasoning: {reasoning}, Saved: {current_result['saved_model_path'] is not None}"
            )
            logger.info(f"Result: Accuracy = {accuracy:.4f}, Accept: {accept}, Reasoning: {reasoning}, Saved: {current_result['saved_model_path'] is not None}")

            if accept and accuracy >= accuracy_threshold:
                logger.info(f"Accuracy threshold {accuracy_threshold} met by an accepted model. Stopping optimization.")
                stop_due_to_threshold = True
                break # Stop evaluating other recommendations in this iteration
        
        previous_results = "\n".join(iteration_summary_for_recommender) # Update with all results from this iteration

        if stop_due_to_threshold:
            break # Stop master iteration loop

    if not stop_due_to_threshold:
        logger.info("Max iterations reached or no more recommendations.")
    
    # Save all results to a JSON file
    results_timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_filename = f"optimization_results_{results_timestamp}.json"
    results_filepath = os.path.join(results_dir, results_filename)
    try:
        with open(results_filepath, "w") as f:
            json.dump(results, f, indent=4)
        logger.info(f"All optimization results saved to {results_filepath}")
    except Exception as e:
        logger.error(f"Failed to save results JSON: {e}")
        
    return results

if __name__ == "__main__":
    # Load Iris dataset
    iris = load_iris()
    X_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    y_series = pd.Series(iris.target)

    # Run optimization
    final_results = run_optimization(X_df, y_series, max_iterations=5, accuracy_threshold=0.97) # Adjusted for testing
    
    logger.info("\nFinal Optimization Summary:")
    if final_results:
        for res in final_results:
            logger.info(
                f"Model: {res['model']}, "
                f"Hyperparameters: {res['hyperparameters']}, "
                f"Accuracy: {res['accuracy']:.4f}, "
                f"Accept: {res['accept']}, "
                # f"Reasoning: {res['reasoning']}, " # Reasoning can be long
                f"Saved Model: {res['saved_model_path']}"
            )
    else:
        logger.info("No results from the optimization process.")