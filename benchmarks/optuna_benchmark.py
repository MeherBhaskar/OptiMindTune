import logging
import pandas as pd
import time
from datetime import datetime
import json
from pathlib import Path
import optuna
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Output directory setup
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
LOGS_DIR = OUTPUT_DIR / "optuna_benchmarks"
CSV_DIR = OUTPUT_DIR / "csv_results"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

def create_model(trial, model_name):
    """Create a model with hyperparameters suggested by Optuna."""
    if model_name == "RandomForestClassifier":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int('n_estimators', 10, 1000),
            max_depth=trial.suggest_int('max_depth', 1, 32),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
            bootstrap=trial.suggest_categorical('bootstrap', [True, False])
        )
    elif model_name == "LogisticRegression":
        return LogisticRegression(
            C=trial.suggest_float('C', 1e-10, 1e10, log=True),
            solver=trial.suggest_categorical('solver', ['lbfgs', 'saga']),
            max_iter=1000
        )
    elif model_name == "SVC":
        return SVC(
            C=trial.suggest_float('C', 1e-10, 1e10, log=True),
            kernel=trial.suggest_categorical('kernel', ['rbf', 'linear']),
            gamma=trial.suggest_float('gamma', 1e-10, 1e10, log=True)
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

def objective(trial, X, y, model_name):
    """Optuna objective function for model optimization."""
    model = create_model(trial, model_name)
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    return scores.mean()

def optimize_model(X, y, model_name, n_trials=10):
    """Run Optuna optimization for a specific model."""
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X, y, model_name), n_trials=n_trials)
    
    return {
        "best_params": study.best_params,
        "best_accuracy": study.best_value,
        "n_trials": len(study.trials),
        "optimization_history": [
            {"number": t.number, "accuracy": t.value, "params": t.params}
            for t in study.trials
        ]
    }

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    models = ["RandomForestClassifier", "LogisticRegression"]#, "SVC"]
    datasets = {
        'iris': load_iris(),
        'wine': load_wine(),
        'breast_cancer': load_breast_cancer()
    }
    
    benchmark_results = {}
    csv_rows = []
    
    for dataset_name, data in datasets.items():
        logger.info(f"\nOptimizing models for {dataset_name} dataset")
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)
        
        dataset_results = {}
        
        for model_name in models:
            logger.info(f"Optimizing {model_name}...")
            start_time = time.time()
            
            try:
                results = optimize_model(X, y, model_name)
                end_time = time.time()
                optimization_time = end_time - start_time
                
                dataset_results[model_name] = {
                    "best_params": results["best_params"],
                    "best_accuracy": results["best_accuracy"],
                    "n_trials": results["n_trials"],
                    "optimization_time": optimization_time,
                    "trials_per_second": results["n_trials"] / optimization_time,
                    "optimization_history": results["optimization_history"]
                }
                
                # Add row for CSV
                csv_rows.append({
                    "timestamp": timestamp,
                    "dataset": dataset_name,
                    "model": model_name,
                    "best_accuracy": results["best_accuracy"],
                    "optimization_time": optimization_time,
                    "trials_per_second": results["n_trials"] / optimization_time,
                    "n_trials": results["n_trials"],
                    "best_parameters": str(results["best_params"])
                })
                
                logger.info(f"Best accuracy: {results['best_accuracy']:.4f}")
                logger.info(f"Best parameters: {results['best_params']}")
                logger.info(f"Optimization time: {optimization_time:.2f} seconds")
                
            except Exception as e:
                logger.error(f"Error optimizing {model_name}: {e}")
                dataset_results[model_name] = {"error": str(e)}
                
                # Add error row for CSV
                csv_rows.append({
                    "timestamp": timestamp,
                    "dataset": dataset_name,
                    "model": model_name,
                    "best_accuracy": None,
                    "optimization_time": None,
                    "trials_per_second": None,
                    "n_trials": None,
                    "best_parameters": f"ERROR: {str(e)}"
                })
        
        benchmark_results[dataset_name] = dataset_results
    
    # Save results to JSON
    output_file = LOGS_DIR / f"optuna_benchmark_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    
    # Save results to CSV
    csv_file = CSV_DIR / f"optuna_benchmark_{timestamp}.csv"
    pd.DataFrame(csv_rows).to_csv(csv_file, index=False)
    logger.info(f"Results saved to CSV: {csv_file}")
    
    # Print final summary
    logger.info("\n=== Optuna Benchmark Summary ===")
    for dataset_name, dataset_results in benchmark_results.items():
        logger.info(f"\n{dataset_name.upper()} Dataset:")
        for model_name, results in dataset_results.items():
            if "error" not in results:
                logger.info(f"\n{model_name}:")
                logger.info(f"Best Accuracy: {results['best_accuracy']:.4f}")
                logger.info(f"Best Parameters: {results['best_params']}")
                logger.info(f"Optimization Time: {results['optimization_time']:.2f} seconds")
                logger.info(f"Trials/Second: {results['trials_per_second']:.2f}")

if __name__ == "__main__":
    main()
