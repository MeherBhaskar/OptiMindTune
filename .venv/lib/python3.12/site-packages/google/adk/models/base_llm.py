# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from abc import abstractmethod
from typing import AsyncGenerator
from typing import TYPE_CHECKING

from pydantic import BaseModel
from pydantic import ConfigDict

from .base_llm_connection import BaseLlmConnection

if TYPE_CHECKING:
  from .llm_request import LlmRequest
  from .llm_response import LlmResponse


class BaseLlm(BaseModel):
  """The BaseLLM class.

  Attributes:
    model: The name of the LLM, e.g. gemini-1.5-flash or gemini-1.5-flash-001.
    model_config: The model config
  """

  model_config = ConfigDict(
      # This allows us to use arbitrary types in the model. E.g. PIL.Image.
      arbitrary_types_allowed=True,
  )
  """The model config."""

  model: str
  """The name of the LLM, e.g. gemini-1.5-flash or gemini-1.5-flash-001."""

  @classmethod
  def supported_models(cls) -> list[str]:
    """Returns a list of supported models in regex for LlmRegistry."""
    return []

  @abstractmethod
  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    """Generates one content from the given contents and tools.

    Args:
      llm_request: LlmRequest, the request to send to the LLM.
      stream: bool = False, whether to do streaming call.

    Yields:
      a generator of types.Content.

      For non-streaming call, it will only yield one Content.

      For streaming call, it may yield more than one content, but all yielded
      contents should be treated as one content by merging the
      parts list.
    """
    raise NotImplementedError(
        f'Async generation is not supported for {self.model}.'
    )
    yield  # AsyncGenerator requires a yield statement in function body.

  def connect(self, llm_request: LlmRequest) -> BaseLlmConnection:
    """Creates a live connection to the LLM.

    Args:
      llm_request: LlmRequest, the request to send to the LLM.

    Returns:
      BaseLlmConnection, the connection to the LLM.
    """
    raise NotImplementedError(
        f'Live connection is not supported for {self.model}.'
    )
