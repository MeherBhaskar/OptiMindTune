from typing import List, Dict, Any
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from agents.recommender.agent import RecommenderAgent
from agents.evaluator.agent import EvaluatorAgent
import json
import logging
from dotenv import load_dotenv
import os
import joblib
import time
import shutil  # Added for copying files

# Load environment variables
load_dotenv()

# Create module-level logger
logger = logging.getLogger(__name__)

# Model map for re-instantiating models before saving
MODEL_CLASS_MAP = {
    "RandomForestClassifier": RandomForestClassifier,
    "LogisticRegression": LogisticRegression,
    "SVC": SVC
}

def configure_logger(verbose: bool):
    """
    Configure the root logger level.
    If verbose=True, show INFO and above; if False, only WARNING and above.
    """
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger.setLevel(level)

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

def run_optimization(
    X: pd.DataFrame,
    y: pd.Series,
    max_iterations: int = 10,
    accuracy_threshold: float = 0.99,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """Run the multi-agent optimization loop and save artifacts."""
    configure_logger(verbose)
    recommender = RecommenderAgent()
    evaluator = EvaluatorAgent(X, y)
    results: List[Dict[str, Any]] = []
    previous_results = ""
    
    # Prepare artifact folders
    artifacts_dir = "artifacts"
    models_dir = os.path.join(artifacts_dir, "models")
    results_dir = os.path.join(artifacts_dir, "results")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    stop_due_to_threshold = False

    for iteration in range(max_iterations):
        logger.info(f"Iteration {iteration + 1}/{max_iterations}")
        recommendations_str = recommender.recommend(X, y, previous_results)
        recommendations = parse_llm_response(recommendations_str) or []

        # Parse hyperparameters from strings to dicts
        parsed_recommendations = []
        for rec in recommendations:
            model_name = rec.get("model")
            if model_name not in recommender.model_map:
                logger.warning(f"Unsupported model recommended: {model_name}")
                continue

            hyperparams_str = rec.get("hyperparameters", "")
            hp_dict: Dict[str, Any] = {}
            try:
                for pair in hyperparams_str.split(","):
                    if "=" not in pair:
                        continue
                    key, value = pair.split("=", 1)
                    key = key.strip()
                    val = value.strip().strip("'\"")
                    # convert to int or float if possible
                    try:
                        val = int(val)
                    except ValueError:
                        try:
                            val = float(val)
                        except ValueError:
                            pass
                    hp_dict[key] = val
            except Exception as e:
                logger.error(f"Failed parsing hyperparams: {e} — raw: {hyperparams_str}")
                continue

            rec["hyperparameters"] = hp_dict
            parsed_recommendations.append(rec)

        if not parsed_recommendations:
            logger.info("No valid recommendations this iteration.")
            if not results:
                logger.warning("No prior results either; stopping early.")
                break

        iteration_summary: List[str] = []

        for idx, rec in enumerate(parsed_recommendations):
            model_name = rec["model"]
            hyperparams = rec["hyperparameters"]
            logger.info(f"Evaluating {model_name} with {hyperparams}")

            eval_response = evaluator.evaluate_and_suggest(model_name, hyperparams, previous_results)
            eval_result = parse_llm_response(eval_response)
            if not eval_result:
                logger.warning(f"Invalid eval response for {model_name}; skipping.")
                continue

            accuracy = eval_result.get("accuracy", 0.0)
            accept = eval_result.get("accept", False)
            reasoning = eval_result.get("reasoning", "")

            result_record = {
                "iteration": iteration + 1,
                "recommendation_index": idx + 1,
                "model": model_name,
                "hyperparameters": hyperparams,
                "accuracy": accuracy,
                "accept": accept,
                "reasoning": reasoning,
                "saved_model_path": None
            }

            if accept:
                logger.info(f"Accepted {model_name} @ {accuracy:.4f}")
                try:
                    model_cls = MODEL_CLASS_MAP[model_name]
                    # adjust for l1+saga
                    if model_name == "LogisticRegression":
                        if hyperparams.get("solver") == "saga" and hyperparams.get("penalty") == "l1":
                            hyperparams.setdefault("max_iter", 1000)

                    pipeline = Pipeline([
                        ("scaler", StandardScaler()),
                        ("clf", model_cls(**hyperparams))
                    ])
                    pipeline.fit(X, y)

                    ts = time.strftime("%Y%m%d_%H%M%S")
                    fname = f"{model_name}_iter{iteration+1}_rec{idx+1}_{ts}.joblib"
                    fpath = os.path.join(models_dir, fname)
                    joblib.dump(pipeline, fpath)
                    result_record["saved_model_path"] = fpath
                    logger.info(f"Model saved to {fpath}")
                except Exception as e:
                    logger.error(f"Error saving model: {e}")

            results.append(result_record)

            iteration_summary.append(
                f"{model_name} with {hyperparams} → acc={accuracy:.4f}, "
                f"accept={accept}, saved={bool(result_record['saved_model_path'])}"
            )
            logger.info("Detailed: " + iteration_summary[-1])

            if accept and accuracy >= accuracy_threshold:
                logger.info(f"Threshold {accuracy_threshold} reached; stopping.")
                stop_due_to_threshold = True
                break

        previous_results = "\n".join(iteration_summary)
        if stop_due_to_threshold:
            break

    if not stop_due_to_threshold:
        logger.info("Finished all iterations or no more valid recs.")

    # Save results JSON
    ts_all = time.strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"optimization_results_{ts_all}.json")
    try:
        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Results JSON saved to {results_file}")
    except Exception as e:
        logger.error(f"Failed writing results JSON: {e}")

    return results

if __name__ == "__main__":
    # Load data
    iris = load_iris()
    X_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    y_series = pd.Series(iris.target)

    # Run with verbose=True or False
    final_results = run_optimization(
        X_df,
        y_series,
        max_iterations=5,
        accuracy_threshold=0.97,
        verbose=True  # ← set to True to see full log, False to suppress intermediates
    )

    # Always show final summary
    # Temporarily bump logger to INFO if it was suppressed
    logger.setLevel(logging.INFO)
    logger.info("\n=== Final Optimization Summary ===")
    if final_results:
        for r in final_results:
            logger.info(
                f"Iter{r['iteration']}-Rec{r['recommendation_index']}: "
                f"{r['model']} @ {r['accuracy']:.4f}, accept={r['accept']}, "
                f"saved={bool(r['saved_model_path'])}"
            )
    else:
        logger.info("No results returned.")

    # Select best model
    best_info = None
    best_acc = -1.0
    for r in final_results:
        if r['accept'] and r['accuracy'] > best_acc and r['saved_model_path'] and os.path.exists(r['saved_model_path']):
            best_acc = r['accuracy']
            best_info = r

    if best_info:
        logger.info("\n--- Best Model ---")
        logger.info(f"{best_info['model']} @ {best_info['accuracy']:.4f}")
        logger.info(f"Params: {best_info['hyperparameters']}")
        logger.info(f"Path: {best_info['saved_model_path']}")

        best_dir = os.path.join("artifacts", "best_model")
        os.makedirs(best_dir, exist_ok=True)
        dest = os.path.join(best_dir, "best_model.joblib")
        try:
            shutil.copy(best_info['saved_model_path'], dest)
            logger.info(f"Copied to {dest}")
        except Exception as e:
            logger.error(f"Error copying best: {e}")

        details_path = os.path.join(best_dir, "best_model_details.json")
        try:
            with open(details_path, "w") as f:
                json.dump(best_info, f, indent=4)
            logger.info(f"Details saved to {details_path}")
        except Exception as e:
            logger.error(f"Error saving details: {e}")
    else:
        logger.info("No acceptable model to mark as best.")
