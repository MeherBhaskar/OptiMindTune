from typing import Dict, Any
import pandas as pd
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from sklearn.datasets import load_iris
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

class ModelRecommendationAgent:
    def __init__(self):
        """Initialize the LangChain agent with Mistral-7B."""
        self.llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            temperature=0.7,
            max_length=512,
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )
        self.prompt_template = PromptTemplate(
            input_variables=["dataset_info"],
            template="""
You are an expert AI agent for AutoML. Given the following dataset metadata, recommend 1-3 scikit-learn classification models and suggest key hyperparameters to tune. Provide a brief reasoning for each recommendation.

Dataset Metadata:
{dataset_info}

Output format:
- Model: [Model Name]
  - Hyperparameters: [Key hyperparameters to tune]
  - Reasoning: [Why this model is suitable]
"""
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    def get_dataset_metadata(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Extract metadata from the dataset."""
        return {
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "n_classes": len(y.unique()),
            "class_balance": y.value_counts(normalize=True).to_dict(),
        }

    def recommend_models(self, X: pd.DataFrame, y: pd.Series) -> str:
        """Recommend models based on dataset metadata."""
        metadata = self.get_dataset_metadata(X, y)
        dataset_info = "\n".join([f"{k}: {v}" for k, v in metadata.items()])
        response = self.chain.invoke({"dataset_info": dataset_info})
        return response.get("text", "")

# Example usage
if __name__ == "__main__":
    # Load a sample dataset (Iris)
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)

    # Initialize agent and get recommendations
    agent = ModelRecommendationAgent()
    recommendations = agent.recommend_models(X, y)
    print("Model Recommendations:\n", recommendations)