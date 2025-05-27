"""
Orchestrator: Manages the main optimization workflow
"""

import logging
import json
from typing import Dict, Any, Optional, Tuple
import pandas as pd
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

from agents.recommender import RecommenderAgent
from agents.evaluation import EvaluationAgent
from agents.decision import DecisionAgent
from core.session_manager import SessionManager
from core.agent_runner import AgentRunner
from config import OptiMindConfig, OptimizationComplete

logger = logging.getLogger(__name__)


class OptimizationOrchestrator:
    """Orchestrates the optimization process across multiple agents."""
    
    def __init__(self, config: OptiMindConfig, results_manager):
        self.config = config
        self.results_manager = results_manager
        self.session_manager = SessionManager(config)
        self.agent_runner = AgentRunner(config)
        
    def optimize_dataset(self, dataset_name: str, X: pd.DataFrame, y: pd.Series, timestamp: str) -> Dict[str, Any]:
        """
        Optimize a single dataset using the agent-based approach.
        
        Args:
            dataset_name: Name of the dataset
            X: Feature matrix
            y: Target vector
            timestamp: Unique timestamp for this run
            
        Returns:
            Dictionary containing optimization results
        """
        logger.info(f"Initializing optimization for {dataset_name}")
        
        # Initialize sessions and agents for this dataset
        sessions = self.session_manager.create_dataset_sessions(dataset_name)
        agents = self._create_agents(X, y)
        runners = self._create_runners(agents)
        
        # Track optimization state
        best_model = None
        best_accuracy = 0.0
        iteration = 0
        
        try:
            for iteration in range(self.config.max_iterations):
                logger.info(f"Starting iteration {iteration + 1}/{self.config.max_iterations}")
                
                # Get recommendations
                recommendations = self._get_recommendations(
                    X, y, sessions["recommender"], runners["recommender"], 
                    dataset_name, iteration, timestamp
                )
                
                if not recommendations:
                    logger.warning("No valid recommendations received, continuing...")
                    continue
                
                # Process recommendations
                best_iteration_result = self._process_recommendations(
                    recommendations, runners, sessions, dataset_name, 
                    iteration, timestamp
                )
                
                # Update best model if improved
                if best_iteration_result and best_iteration_result["accuracy"] > best_accuracy:
                    best_accuracy = best_iteration_result["accuracy"]
                    best_model = best_iteration_result
                    logger.info(f"New best model found: {best_model['model']} with accuracy {best_accuracy:.4f}")
                
                # Check early stopping criteria
                if self._should_stop_optimization(best_accuracy, iteration):
                    break
                    
        except OptimizationComplete:
            logger.info("Optimization completed by agent decision")
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            
        return {
            "best_model": best_model["model"] if best_model else None,
            "best_hyperparameters": best_model["hyperparameters"] if best_model else None,
            "best_accuracy": best_accuracy,
            "total_iterations": iteration + 1,
            "final_result": best_model
        }
    
    def _create_agents(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Create agent instances for this optimization run."""
        return {
            "recommender": RecommenderAgent(model=self.config.model_name),
            "evaluator": EvaluationAgent(X, y, model=self.config.model_name),
            "decision": DecisionAgent(model=self.config.model_name)
        }
    
    def _create_runners(self, agents: Dict[str, Any]) -> Dict[str, Runner]:
        """Create runner instances for each agent."""
        return {
            name: Runner(
                app_name=self.config.app_name,
                agent=agent,
                session_service=self.session_manager.session_service
            )
            for name, agent in agents.items()
        }
    
    def _get_recommendations(self, X: pd.DataFrame, y: pd.Series, session, runner: Runner, 
                           dataset_name: str, iteration: int, timestamp: str) -> list:
        """Get model recommendations from the recommender agent."""
        previous_results = session.state.get("evaluation_history", [])
        prompt = self._build_recommendation_prompt(X, y, previous_results)
        
        message = types.Content(role="user", parts=[types.Part(text=prompt)])
        event = self.agent_runner.run_agent(runner, message, f"rec_session_{dataset_name}")
        
        recommendations = self._parse_recommendations(event)
        
        # Log the recommendation interaction
        self.results_manager.save_conversation({
            "iteration": iteration,
            "agent": "recommender",
            "dataset": dataset_name,
            "input": prompt,
            "output": recommendations,
            "status": "success" if recommendations else "failed"
        }, timestamp)
        
        return recommendations
    
    def _process_recommendations(self, recommendations: list, runners: Dict[str, Runner], 
                               sessions: Dict, dataset_name: str, iteration: int, 
                               timestamp: str) -> Optional[Dict[str, Any]]:
        """Process all recommendations and return the best one."""
        best_result = None
        best_accuracy = 0.0
        
        for rec in recommendations:
            model_name = rec.get('model')
            hyperparameters = rec.get('hyperparameters')
            reasoning = rec.get('reasoning')
            
            logger.info(f"Processing: {model_name} with {hyperparameters}")
            logger.info(f"Reasoning: {reasoning}")
            
            # Evaluate model
            eval_result = self._evaluate_recommendation(
                rec, runners["evaluator"], dataset_name, iteration, timestamp
            )
            
            if not eval_result.get("success"):
                logger.warning(f"Evaluation failed for {model_name}")
                continue
            
            # Make decision
            decision = self._make_decision(
                rec, eval_result, sessions["recommender"].state.get("evaluation_history", []),
                runners["decision"], dataset_name, iteration, timestamp
            )
            
            if decision and decision.get("accept"):
                current_accuracy = eval_result.get("accuracy", 0.0)
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    best_result = {
                        "model": model_name,
                        "hyperparameters": hyperparameters,
                        "accuracy": current_accuracy,
                        "reasoning": decision.get("reasoning", ""),
                        "accept": True
                    }
        
        # Update session state with best result
        if best_result:
            if "evaluation_history" not in sessions["recommender"].state:
                sessions["recommender"].state["evaluation_history"] = []
            sessions["recommender"].state["evaluation_history"].append(best_result)
        
        return best_result
    
    def _evaluate_recommendation(self, recommendation: Dict, runner: Runner, 
                               dataset_name: str, iteration: int, timestamp: str) -> Dict[str, Any]:
        """Evaluate a single model recommendation."""
        eval_input = f"Evaluate {recommendation['model']} with {recommendation['hyperparameters']}"
        message = types.Content(role="user", parts=[types.Part(text=eval_input)])
        
        event = self.agent_runner.run_agent(runner, message, f"eval_session_{dataset_name}")
        
        result = {"accuracy": 0.0, "success": False}
        if event:
            try:
                result = json.loads(event.content.parts[0].text)
                result["success"] = True
                logger.info(f"Evaluation result: {result}")
            except Exception as e:
                logger.error(f"Failed to parse evaluation result: {e}")
        
        # Log evaluation
        self.results_manager.save_conversation({
            "iteration": iteration,
            "agent": "evaluator",
            "model": recommendation["model"],
            "input": eval_input,
            "output": result,
            "status": "success" if result.get("success") else "failed"
        }, timestamp)
        
        return result
    
    def _make_decision(self, recommendation: Dict, eval_result: Dict, previous_results: list, 
                      runner: Runner, dataset_name: str, iteration: int, timestamp: str) -> Optional[Dict]:
        """Make a decision about whether to accept a model."""
        decision_input = f"""Decide whether to accept {recommendation['model']} with accuracy {eval_result['accuracy']:.4f}.
Previous results: {json.dumps(previous_results)}"""
        
        message = types.Content(role="user", parts=[types.Part(text=decision_input)])
        event = self.agent_runner.run_agent(runner, message, f"dec_session_{dataset_name}")
        
        if not event:
            return None
            
        try:
            decision = json.loads(event.content.parts[0].text)
            
            # Log decision
            self.results_manager.save_conversation({
                "iteration": iteration,
                "agent": "decision",
                "model": recommendation["model"],
                "input": decision_input,
                "output": decision,
                "status": "success"
            }, timestamp)
            
            return decision
        except Exception as e:
            logger.error(f"Failed to parse decision result: {e}")
            return None
    
    def _build_recommendation_prompt(self, X: pd.DataFrame, y: pd.Series, previous_results: list) -> str:
        """Build the prompt for the recommender agent."""
        return f"""
Dataset Metadata:
n_samples: {X.shape[0]}
n_features: {X.shape[1]}
n_classes: {len(y.unique())}
class_balance: {y.value_counts(normalize=True).to_dict()}

Previous Results (if any):
{json.dumps(previous_results)}

Supported Models: RandomForestClassifier, LogisticRegression, SVC

Respond with a JSON object: {{"recommendations": [{{"model": "ModelName", "hyperparameters": "param1=value1, param2=value2", "reasoning": "Your reasoning"}}]}}
"""
    
    def _parse_recommendations(self, event) -> list:
        """Parse recommendations from agent response."""
        if not event:
            return []
        try:
            content = event.content.parts[0].text
            parsed = json.loads(content)
            return parsed.get('recommendations', [])
        except (json.JSONDecodeError, AttributeError, KeyError) as e:
            logger.error(f"Failed to parse recommendations: {e}")
            return []
    
    def _should_stop_optimization(self, best_accuracy: float, iteration: int) -> bool:
        """Determine if optimization should stop early."""
        # Stop if we've reached target accuracy and explored enough
        if best_accuracy >= self.config.target_accuracy:
            if iteration >= self.config.max_iterations * self.config.exploration_ratio:
                logger.info(f"Target accuracy {self.config.target_accuracy} reached, stopping optimization")
                return True
        
        # Stop if accuracy is too low
        if best_accuracy < self.config.min_accuracy and iteration >= 3:
            logger.warning(f"Accuracy {best_accuracy:.4f} below minimum threshold, stopping")
            return True
            
        return False