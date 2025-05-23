"""
Run benchmarking experiments comparing OptiMindTune with other AutoML frameworks.
"""

from sklearn.datasets import load_breast_cancer, load_iris, load_digits
import pandas as pd
import numpy as np
from automl_benchmark import AutoMLBenchmark

def load_datasets():
    """Load benchmark datasets."""
    datasets = {}
    
    # Load iris dataset
    iris = load_iris()
    datasets['iris'] = (iris.data, iris.target)
    
    # Load breast cancer dataset
    cancer = load_breast_cancer()
    datasets['breast_cancer'] = (cancer.data, cancer.target)
    
    # Load digits dataset
    digits = load_digits()
    datasets['digits'] = (digits.data, digits.target)
    
    return datasets

def main():
    # Load datasets
    print("Loading datasets...")
    datasets = load_datasets()
    
    # Initialize benchmark
    print("Initializing benchmark...")
    benchmark = AutoMLBenchmark(
        datasets=datasets,
        time_budget=1800,  # 30 minutes per framework
        metrics=['accuracy', 'f1', 'roc_auc']
    )
    
    # Run benchmark
    print("Running benchmark...")
    results = benchmark.run_benchmark()
    
    # Save results
    results.to_csv('benchmark_results.csv', index=False)
    print("\nResults saved to benchmark_results.csv")
    
    # Plot results
    print("Plotting results...")
    plots = benchmark.plot_results()
    plots.savefig('benchmark_plots.pdf')
    print("Plots saved to benchmark_plots.pdf")
    
    # Print summary
    print("\nBenchmark Summary:")
    print("=================")
    print("\nMean Performance by Framework:")
    mean_perf = results.groupby('framework')[['accuracy', 'f1', 'roc_auc', 'time']].mean()
    print(mean_perf)
    
    print("\nBest Framework by Dataset:")
    for metric in ['accuracy', 'f1', 'roc_auc']:
        best_by_dataset = results.loc[results.groupby('dataset')[metric].idxmax()]
        print(f"\n{metric.upper()}:")
        for _, row in best_by_dataset.iterrows():
            print(f"{row['dataset']}: {row['framework']} ({row[metric]:.3f})")

if __name__ == "__main__":
    main()
