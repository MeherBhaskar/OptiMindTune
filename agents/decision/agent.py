from google.adk.agents import Agent
from pydantic import BaseModel, Field

class EvaluationOutput(BaseModel):
    accuracy: float = Field(description="The accuracy of the model")
    accept: bool = Field(description="Whether to accept the model")
    reasoning: str = Field(description="Reasoning for the decision")

class DecisionAgent(Agent):
    def __init__(self):
        super().__init__(
            name="decision",
            model="gemini-2.0-flash",
            instruction="You are a decision agent. Given a model name, hyperparameters, accuracy, and previous results, decide whether to accept the model or recommend new ones. Respond with a JSON object: {\"accuracy\": float, \"accept\": boolean, \"reasoning\": \"Your reasoning\"}",
            output_schema=EvaluationOutput,
            output_key="evaluation_result",
            disallow_transfer_to_parent=True,
            disallow_transfer_to_peers=True
        )