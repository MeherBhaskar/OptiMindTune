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

"""Sequential agent implementation."""

from __future__ import annotations

from typing import AsyncGenerator

from typing_extensions import override

from ..agents.invocation_context import InvocationContext
from ..events.event import Event
from .base_agent import BaseAgent


class SequentialAgent(BaseAgent):
  """A shell agent that run its sub-agents in sequence."""

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    for sub_agent in self.sub_agents:
      async for event in sub_agent.run_async(ctx):
        yield event

  @override
  async def _run_live_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    for sub_agent in self.sub_agents:
      async for event in sub_agent.run_live(ctx):
        yield event
