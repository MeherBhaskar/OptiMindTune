from google.adk.agents import Agent
from utils.logger import ConversationLogger
from pydantic import BaseModel, PrivateAttr
from typing import Optional

class BaseLoggingAgent(Agent, BaseModel):
    _logger: Optional[ConversationLogger] = PrivateAttr(default=None)
    _name: str = PrivateAttr()

    def __init__(self, name: str, **kwargs):
        Agent.__init__(self, name=name, **kwargs)
        BaseModel.__init__(self)
        self._name = name
        self._logger = ConversationLogger(name)

    async def log_interaction(self, timestamp: str, iteration: int, 
                            input_data: str, output_data: dict, status: str = "success"):
        if self._logger:
            await self._logger.log_interaction(timestamp, iteration, input_data, output_data, status)
