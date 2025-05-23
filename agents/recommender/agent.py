from google.adk.agents import Agent
from pydantic import BaseModel, Field
from typing import List

class RecommendationItem(BaseModel):
    model: str = Field(description="The name of the model")
    hyperparameters: str = Field(description="The hyperparameters for the model")
    reasoning: str = Field(description="The reasoning for recommending this model")

class RecommendationOutput(BaseModel):
    recommendations: List[RecommendationItem] = Field(description="List of recommended models")

class RecommenderAgent(Agent):
    def __init__(self, model="gemini-2.0-flash"):
        super().__init__(
            name="recommender",
            model=model,
            instruction="You are an expert AI agent for AutoML. Given the dataset metadata and previous evaluation results, recommend 1-2 scikit-learn classification models and specific hyperparameter configurations to try next. If no previous results exist, suggest initial models and hyperparameters. Provide a brief reasoning for each recommendation. Respond with a JSON object: {\"recommendations\": [{\"model\": \"ModelName\", \"hyperparameters\": \"param1=value1, param2=value2 param3=value3\", \"reasoning\": \"Your reasoning\"}]}",
            output_schema=RecommendationOutput,
            output_key="recommendations",
            disallow_transfer_to_parent=True,
            disallow_transfer_to_peers=True
        )