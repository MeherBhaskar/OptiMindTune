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

import inspect
from typing import Any
from typing import Callable
from typing import Optional

from google.genai import types
from typing_extensions import override

from ._automatic_function_calling_util import build_function_declaration
from .base_tool import BaseTool
from .tool_context import ToolContext


class FunctionTool(BaseTool):
  """A tool that wraps a user-defined Python function.

  Attributes:
    func: The function to wrap.
  """

  def __init__(self, func: Callable[..., Any]):
    super().__init__(name=func.__name__, description=func.__doc__)
    self.func = func

  @override
  def _get_declaration(self) -> Optional[types.FunctionDeclaration]:
    function_decl = types.FunctionDeclaration.model_validate(
        build_function_declaration(
            func=self.func,
            # The model doesn't understand the function context.
            # input_stream is for streaming tool
            ignore_params=['tool_context', 'input_stream'],
            variant=self._api_variant,
        )
    )

    return function_decl

  @override
  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    args_to_call = args.copy()
    signature = inspect.signature(self.func)
    if 'tool_context' in signature.parameters:
      args_to_call['tool_context'] = tool_context

    if inspect.iscoroutinefunction(self.func):
      return await self.func(**args_to_call) or {}
    else:
      return self.func(**args_to_call) or {}

  # TODO(hangfei): fix call live for function stream.
  async def _call_live(
      self,
      *,
      args: dict[str, Any],
      tool_context: ToolContext,
      invocation_context,
  ) -> Any:
    args_to_call = args.copy()
    signature = inspect.signature(self.func)
    if (
        self.name in invocation_context.active_streaming_tools
        and invocation_context.active_streaming_tools[self.name].stream
    ):
      args_to_call['input_stream'] = invocation_context.active_streaming_tools[
          self.name
      ].stream
    if 'tool_context' in signature.parameters:
      args_to_call['tool_context'] = tool_context
    async for item in self.func(**args_to_call):
      yield item
