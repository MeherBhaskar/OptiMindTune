from typing import List, Dict, Any
import pandas as pd
from sklearn.datasets import load_iris
from agents.recommender.agent import RecommenderAgent
from agents.evaluator.agent import EvaluatorAgent
import json
import logging
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    """Run the multi-agent optimization loop."""
    recommender = RecommenderAgent()
    evaluator = EvaluatorAgent(X, y)
    results = []
    previous_results = ""

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
            if rec.get("model") not in recommender.model_map:
                logger.warning(f"Unsupported model: {rec.get('model')}")
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
                                pass
                        hyperparams_dict[key] = value
            except Exception as e:
                logger.error(f"Failed to parse hyperparameters: {e}, Hyperparameters: {hyperparams_str}")
                continue
            rec["hyperparameters"] = hyperparams_dict
            parsed_recommendations.append(rec)

        if not parsed_recommendations:
            logger.info("No valid recommendations received. Stopping.")
            break

        # Evaluate each recommendation
        iteration_results = []
        for rec in parsed_recommendations:
            model_name = rec["model"]
            hyperparameters = rec["hyperparameters"]
            logger.info(f"Evaluating {model_name} with {hyperparameters}")

            # Evaluate and get suggestion
            eval_response = evaluator.evaluate_and_suggest(model_name, hyperparameters, previous_results)
            eval_result = parse_llm_response(eval_response)
            if eval_result is None:
                continue
            accuracy = eval_result.get("accuracy", 0.0)
            accept = eval_result.get("accept", False)
            reasoning = eval_result.get("reasoning", "No reasoning provided")
            result = {
                "model": model_name,
                "hyperparameters": hyperparameters,
                "accuracy": accuracy,
                "accept": accept,
                "reasoning": reasoning
            }
            results.append(result)
            iteration_results.append(
                f"Model: {model_name}, Hyperparameters: {hyperparameters}, Accuracy: {accuracy:.4f}, Accept: {accept}, Reasoning: {reasoning}"
            )
            logger.info(f"Result: Accuracy = {accuracy:.4f}, Accept: {accept}, Reasoning: {reasoning}")

            # If accepted and meets threshold, stop
            if accept and accuracy >= accuracy_threshold:
                logger.info(f"Accuracy threshold {accuracy_threshold} met. Stopping.")
                return results

        # Update previous results
        previous_results = "\n".join(iteration_results)

    logger.info("Max iterations reached.")
    return results

if __name__ == "__main__":
    # Load Iris dataset
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)

    # Run optimization
    results = run_optimization(X, y)
    logger.info("\nFinal Results:")
    for result in results:
        logger.info(f"Model: {result['model']}, Hyperparameters: {result['hyperparameters']}, Accuracy: {result['accuracy']:.4f}, Accept: {result['accept']}, Reasoning: {result['reasoning']}")