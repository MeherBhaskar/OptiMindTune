"""
Results Manager: Handles logging, conversation tracking, and result output
"""

import json
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class ConfigEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling Path objects and other special types."""
    
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


class ResultsManager:
    """Manages all result storage, logging, and output generation."""
    
    def __init__(self, logs_dir: Path, csv_dir: Path, timestamp: str):
        self.logs_dir = Path(logs_dir)
        self.csv_dir = Path(csv_dir)
        self.timestamp = timestamp
        self.csv_rows = []
        
        # Ensure directories exist
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize conversation log file
        self.conversation_file = self.logs_dir / f"conversation_{timestamp}.json"
        self._initialize_conversation_log()
    
    def _initialize_conversation_log(self):
        """Initialize the conversation log file."""
        initial_data = {
            "metadata": {
                "timestamp": self.timestamp,
                "created_at": datetime.now().isoformat(),
                "version": "1.0"
            },
            "interactions": [],
            "datasets": {}
        }
        
        with open(self.conversation_file, 'w') as f:
            json.dump(initial_data, f, indent=2, cls=ConfigEncoder)
    
    def save_conversation(self, interaction_data: Dict[str, Any], timestamp: str):
        """
        Save a single conversation interaction.
        
        Args:
            interaction_data: Data about the interaction
            timestamp: Timestamp for this run
        """
        try:
            # Load existing data
            with open(self.conversation_file, 'r') as f:
                conversation_data = json.load(f)
            
            # Add new interaction with timestamp
            interaction_data["timestamp"] = datetime.now().isoformat()
            conversation_data["interactions"].append(interaction_data)
            conversation_data["last_updated"] = datetime.now().isoformat()
            
            # Save back to file
            with open(self.conversation_file, 'w') as f:
                json.dump(conversation_data, f, indent=2, cls=ConfigEncoder)
                
        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")
    
    def save_dataset_metadata(self, dataset_name: str, metadata: Dict[str, Any]):
        """
        Save metadata for a completed dataset optimization.
        
        Args:
            dataset_name: Name of the dataset
            metadata: Metadata dictionary
        """
        try:
            with open(self.conversation_file, 'r') as f:
                conversation_data = json.load(f)
            
            # Ensure datasets section exists
            if "datasets" not in conversation_data:
                conversation_data["datasets"] = {}
            
            # Add dataset metadata
            conversation_data["datasets"][dataset_name] = {
                **metadata,
                "completed_at": datetime.now().isoformat()
            }
            
            # Save back to file
            with open(self.conversation_file, 'w') as f:
                json.dump(conversation_data, f, indent=2, cls=ConfigEncoder)
                
        except Exception as e:
            logger.error(f"Failed to save dataset metadata: {e}")
    
    def add_csv_row(self, dataset_name: str, result: Dict[str, Any], optimization_time: float):
        """
        Add a row to the CSV results.
        
        Args:
            dataset_name: Name of the dataset
            result: Optimization result
            optimization_time: Time taken for optimization
        """
        csv_row = {
            "timestamp": self.timestamp,
            "dataset": dataset_name,
            "optimizer": "LLM",
            "model": result.get("best_model"),
            "best_accuracy": result.get("best_accuracy", 0.0),
            "optimization_time": optimization_time,
            "iterations_per_second": result.get("total_iterations", 0) / max(optimization_time, 0.001),
            "total_iterations": result.get("total_iterations", 0),
            "hyperparameters": str(result.get("best_hyperparameters", ""))
        }
        
        self.csv_rows.append(csv_row)
    
    def save_benchmark_summary(self, benchmark_results: Dict[str, Any]):
        """
        Save the final benchmark summary to CSV and update conversation log.
        
        Args:
            benchmark_results: Dictionary of all benchmark results
        """
        # Add rows to CSV for each dataset
        for dataset_name, result in benchmark_results.items():
            if "error" not in result:  # Only add successful results
                self.add_csv_row(dataset_name, result, result.get("optimization_time", 0))
        
        # Save CSV file
        if self.csv_rows:
            csv_file = self.csv_dir / f"optimind_benchmark_{self.timestamp}.csv"
            pd.DataFrame(self.csv_rows).to_csv(csv_file, index=False)
            logger.info(f"Results saved to CSV: {csv_file}")
        
        # Update conversation log with summary
        try:
            with open(self.conversation_file, 'r') as f:
                conversation_data = json.load(f)
            
            conversation_data["benchmark_summary"] = {
                "results": benchmark_results,
                "total_datasets": len(benchmark_results),
                "successful_optimizations": len([r for r in benchmark_results.values() if "error" not in r]),
                "completed_at": datetime.now().isoformat()
            }
            
            with open(self.conversation_file, 'w') as f:
                json.dump(conversation_data, f, indent=2, cls=ConfigEncoder)
                
        except Exception as e:
            logger.error(f"Failed to save benchmark summary: {e}")
    
    def print_summary(self, benchmark_results: Dict[str, Any]):
        """
        Print a formatted summary of benchmark results.
        
        Args:
            benchmark_results: Dictionary of all benchmark results
        """
        logger.info("\n" + "="*60)
        logger.info("OPTIMIND BENCHMARK SUMMARY")
        logger.info("="*60)
        
        successful_results = {k: v for k, v in benchmark_results.items() if "error" not in v}
        failed_results = {k: v for k, v in benchmark_results.items() if "error" in v}
        
        logger.info(f"Total Datasets: {len(benchmark_results)}")
        logger.info(f"Successful Optimizations: {len(successful_results)}")
        logger.info(f"Failed Optimizations: {len(failed_results)}")
        
        if failed_results:
            logger.info("\nFailed Optimizations:")
            for dataset_name, result in failed_results.items():
                logger.info(f"  - {dataset_name}: {result['error']}")
        
        if successful_results:
            logger.info("\nSuccessful Optimizations:")
            logger.info("-" * 60)
            
            for dataset_name, result in successful_results.items():
                logger.info(f"\n{dataset_name.upper()} Dataset:")
                logger.info(f"  Best Model: {result.get('best_model', 'N/A')}")
                logger.info(f"  Best Hyperparameters: {result.get('best_hyperparameters', 'N/A')}")
                logger.info(f"  Best Accuracy: {result.get('best_accuracy', 0):.4f}")
                logger.info(f"  Total Iterations: {result.get('total_iterations', 0)}")
                logger.info(f"  Optimization Time: {result.get('optimization_time', 0):.2f} seconds")
                logger.info(f"  Iterations/Second: {result.get('iterations_per_second', 0):.2f}")
        
        # Overall statistics
        if successful_results:
            avg_accuracy = sum(r.get('best_accuracy', 0) for r in successful_results.values()) / len(successful_results)
            total_time = sum(r.get('optimization_time', 0) for r in successful_results.values())
            total_iterations = sum(r.get('total_iterations', 0) for r in successful_results.values())
            
            logger.info("\nOverall Statistics:")
            logger.info(f"  Average Best Accuracy: {avg_accuracy:.4f}")
            logger.info(f"  Total Optimization Time: {total_time:.2f} seconds")
            logger.info(f"  Total Iterations: {total_iterations}")
            logger.info(f"  Average Iterations per Dataset: {total_iterations / len(successful_results):.1f}")
        
        logger.info("\n" + "="*60)
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the conversation log.
        
        Returns:
            Dictionary with conversation statistics
        """
        try:
            with open(self.conversation_file, 'r') as f:
                conversation_data = json.load(f)
            
            interactions = conversation_data.get("interactions", [])
            
            return {
                "total_interactions": len(interactions),
                "agents_used": list(set(i.get("agent") for i in interactions if i.get("agent"))),
                "datasets_processed": list(set(i.get("dataset") for i in interactions if i.get("dataset"))),
                "success_rate": len([i for i in interactions if i.get("status") == "success"]) / max(len(interactions), 1),
                "file_size_kb": self.conversation_file.stat().st_size / 1024
            }
            
        except Exception as e:
            logger.error(f"Failed to get conversation stats: {e}")
            return {}