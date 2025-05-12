from typing import List, Dict, Any
import pandas as pd
from sklearn.datasets import load_iris
from optimindtune.agent import RecommenderAgent, EvaluatorAgent

def run_optimization(X: pd.DataFrame, y: pd.Series, max_iterations: int = 10, accuracy_threshold: float = 0.99) -> List[Dict[str, Any]]:
    """Run the multi-agent optimization loop."""
    recommender = RecommenderAgent()
    evaluator = EvaluatorAgent()
    results = []
    previous_results = ""

    for iteration in range(max_iterations):
        print(f"\nIteration {iteration + 1}/{max_iterations}")

        # Get recommendations
        recommendations = recommender.recommend(X, y, previous_results)
        if not recommendations:
            print("No valid recommendations received. Stopping.")
            break

        # Evaluate each recommendation
        iteration_results = []
        for rec in recommendations:
            model_name = rec["model"]
            hyperparameters = rec["hyperparameters"]
            print(f"Evaluating {model_name} with {hyperparameters}")

            try:
                accuracy, result = evaluator.evaluate(
                    X, y, model_name, hyperparameters, recommender.model_map
                )
                results.append(result)
                iteration_results.append(
                    f"Model: {model_name}, Hyperparameters: {hyperparameters}, Accuracy: {accuracy:.4f}"
                )
                print(f"Result: Accuracy = {accuracy:.4f}")

                # Check if threshold is met
                if accuracy >= accuracy_threshold:
                    print(f"Accuracy threshold {accuracy_threshold} met. Stopping.")
                    return results
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                continue

        # Update previous results for the next iteration
        previous_results = "\n".join(iteration_results)

    print("Max iterations reached.")
    return results

if __name__ == "__main__":
    # Load Iris dataset
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)

    # Run optimization
    results = run_optimization(X, y)
    print("\nFinal Results:")
    for result in results:
        print(f"Model: {result['model']}, Hyperparameters: {result['hyperparameters']}, Accuracy: {result['accuracy']:.4f}")