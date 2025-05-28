"""
Bayesian Optimization benchmark script for OptiMindTune.
Runs Bayesian optimization on the same datasets and models for comparison.
"""
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
import time
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Output directories
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
CSV_DIR = OUTPUT_DIR / "bayesian_benchmarks"
CSV_DIR.mkdir(parents=True, exist_ok=True)

class BayesianOptimizer:
    def __init__(self, n_iter=50):
        self.n_iter = n_iter
        self.search_spaces = {
            'RandomForestClassifier': {
                'n_estimators': Integer(10, 100),
                'max_depth': Integer(2, 20),
                'min_samples_split': Integer(2, 20),
                'min_samples_leaf': Integer(1, 10)
            },
            'LogisticRegression': {
                'C': Real(1e-4, 1e4, prior='log-uniform'),
                'penalty': Categorical(['l1', 'l2']),
                'solver': Categorical(['liblinear', 'saga'])
            },
            'SVC': {
                'C': Real(1e-4, 1e4, prior='log-uniform'),
                'gamma': Real(1e-6, 1e1, prior='log-uniform'),
                'kernel': Categorical(['rbf', 'linear'])
            }
        }
        self.model_classes = {
            'RandomForestClassifier': RandomForestClassifier,
            'LogisticRegression': LogisticRegression,
            'SVC': SVC
        }
    
    def optimize(self, X, y, model_name):
        """Run Bayesian optimization for a specific model"""
        if model_name not in self.search_spaces:
            raise ValueError(f"Model {model_name} not supported")
            
        model_class = self.model_classes[model_name]
        search_space = self.search_spaces[model_name]
        
        start_time = time.time()
        
        opt = BayesSearchCV(
            model_class(),
            search_space,
            n_iter=self.n_iter,
            cv=5,
            n_jobs=-1,
            scoring='accuracy',
            random_state=42
        )
        
        opt.fit(X, y)
        
        optimization_time = time.time() - start_time
        
        return {
            'model': model_name,
            'best_params': opt.best_params_,
            'best_score': opt.best_score_,
            'all_scores': opt.cv_results_['mean_test_score'].tolist(),
            'n_iterations': len(opt.cv_results_['mean_test_score']),
            'optimization_time': optimization_time,
            'iterations_per_second': len(opt.cv_results_['mean_test_score']) / optimization_time
        }

def run_bayesian_benchmark(n_iter=50):
    """Run Bayesian optimization benchmarks on all datasets"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load all datasets
    datasets = {
        'breast_cancer': load_breast_cancer(),
        'iris': load_iris(),
        'wine': load_wine(),
    }

    csv_rows = []
    benchmark_results = {}

    for dataset_name, data in datasets.items():
        logger.info(f"\nStarting Bayesian optimization for {dataset_name} dataset")
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)

        bayes_opt = BayesianOptimizer(n_iter=n_iter)
        dataset_results = {}
        
        for model_name in ['RandomForestClassifier', 'LogisticRegression', 'SVC']:
            try:
                logger.info(f"Optimizing {model_name}...")
                result = bayes_opt.optimize(X, y, model_name)
                dataset_results[model_name] = result
                
                # Add row for CSV
                csv_rows.append({
                    "timestamp": timestamp,
                    "dataset": dataset_name,
                    "optimizer": "Bayesian",
                    "model": model_name,
                    "best_accuracy": result["best_score"],
                    "optimization_time": result["optimization_time"],
                    "iterations_per_second": result["iterations_per_second"],
                    "total_iterations": result["n_iterations"],
                    "hyperparameters": str(result["best_params"])
                })
                
                logger.info(f"Best accuracy: {result['best_score']:.4f}")
                logger.info(f"Best parameters: {result['best_params']}")
                logger.info(f"Optimization time: {result['optimization_time']:.2f}s")
                
            except Exception as e:
                logger.error(f"Optimization failed for {model_name}: {e}")
        
        benchmark_results[dataset_name] = dataset_results

    # Save results to CSV
    csv_file = CSV_DIR / f"bayesian_benchmark_{timestamp}.csv"
    pd.DataFrame(csv_rows).to_csv(csv_file, index=False)
    logger.info(f"\nResults saved to CSV: {csv_file}")

    # Print final summary
    logger.info("\n=== Bayesian Optimization Benchmark Summary ===")
    for dataset_name, results in benchmark_results.items():
        logger.info(f"\n{dataset_name.upper()} Dataset:")
        for model_name, result in results.items():
            logger.info(f"\n{model_name}:")
            logger.info(f"Best Accuracy: {result['best_score']:.4f}")
            logger.info(f"Optimization Time: {result['optimization_time']:.2f} seconds")
            logger.info(f"Iterations/Second: {result['iterations_per_second']:.2f}")
            logger.info(f"Best Parameters: {result['best_params']}")

if __name__ == "__main__":
    run_bayesian_benchmark()
