from typing import Dict, Any, List
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecommenderAgent:
    def __init__(self, model_name: str = "t5-base"):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            logger.info(f"Model '{model_name}' loaded successfully on {self.device}.")
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}")
            raise

    def get_dataset_metadata(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        return {
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "n_classes": len(y.unique()),
            "class_balance": y.value_counts(normalize=True).to_dict(),
        }

    def recommend(self, X: pd.DataFrame, y: pd.Series, previous_results: str = "") -> List[str]:
        metadata = self.get_dataset_metadata(X, y)
        dataset_info = "\n".join([f"{k}: {v}" for k, v in metadata.items()])

        prompt = f"""
You are an expert AutoML agent.
Given the dataset metadata and previous evaluation results, suggest the next model and hyperparameters to try using the following format:

run_model("ModelName", param1=value1, param2=value2)

Supported Models: RandomForestClassifier, LogisticRegression, SVC

Dataset Metadata:
{dataset_info}

Previous Results:
{previous_results}

Only respond with the Python function call in the above format. Do not include any other text, explanations, or metadata.
"""

        test_input = f"Please provide the next recommendation based on the following data:\n{prompt}"

        try:
            # Tokenize the input and generate output
            inputs = self.tokenizer(test_input, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs)

            # Decode the model output
            response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Check if the response is empty
            if not response_text.strip():
                logger.error("Model output is empty!")
            else:
                logger.info(f"Model Output: {response_text}")

            return [response_text]
        except Exception as e:
            logger.error(f"Model inference failed: {e}")
            return []

# Example usage
if __name__ == "__main__":
    # Dummy data
    X = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [5, 4, 3, 2, 1]
    })
    y = pd.Series([0, 1, 0, 1, 0])

    # Initialize the recommender agent
    agent = RecommenderAgent(model_name="t5-base")

    # Get recommendations based on the dataset
    recommendations = agent.recommend(X, y)
    print("Recommendations:", recommendations)
