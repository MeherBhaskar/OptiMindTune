"""
OptiMind: AI-Powered Model Optimization System
Main entry point for the optimization benchmark.
"""

import logging
import time
from datetime import datetime
from pathlib import Path

from core.orchestrator import OptimizationOrchestrator
from core.data_manager import DatasetManager
from core.results_manager import ResultsManager
from config import OptiMindConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main execution function."""
    try:
        # Initialize configuration
        config = OptiMindConfig()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize managers
        data_manager = DatasetManager()
        results_manager = ResultsManager(config.logs_dir, config.csv_dir, timestamp)
        orchestrator = OptimizationOrchestrator(config, results_manager)
        
        logger.info("Starting OptiMind optimization benchmark")
        
        # Load datasets
        datasets = data_manager.load_datasets("iris")
        logger.info(f"Loaded {len(datasets)} datasets: {list(datasets.keys())}")
        
        # Run optimization for each dataset
        benchmark_results = {}
        
        for dataset_name, (X, y) in datasets.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Starting optimization for {dataset_name} dataset")
            logger.info(f"{'='*50}")
            
            start_time = time.time()
            
            try:
                result = orchestrator.optimize_dataset(dataset_name, X, y, timestamp)
                optimization_time = time.time() - start_time
                
                # Store results
                benchmark_results[dataset_name] = {
                    **result,
                    "optimization_time": optimization_time,
                    "iterations_per_second": result["total_iterations"] / optimization_time
                }
                
                logger.info(f"Completed {dataset_name} optimization in {optimization_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Failed to optimize {dataset_name}: {e}")
                benchmark_results[dataset_name] = {
                    "error": str(e),
                    "optimization_time": time.time() - start_time
                }
        
        # Save final results
        results_manager.save_benchmark_summary(benchmark_results)
        results_manager.print_summary(benchmark_results)
        
        logger.info("OptiMind benchmark completed successfully")
        
    except Exception as e:
        logger.error(f"Critical error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()