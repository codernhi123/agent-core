# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Harbor agent that hands the live trial environment to the single-harness backend.

Harbor imports this module by ``import_path``; it requires the ``harbor``
package (Python 3.12+). The DeepAgent itself is built by
``SingleHarnessExecutionBackend`` so it stays identical to the SWE-bench path.
"""

from __future__ import annotations

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from openjiuwen.rsi.harness_rsi.evaluator.harbor_runtime import live_agent


class OpenJiuwenHarborAgent(BaseAgent):
    """Run the openJiuwen single-harness DeepAgent inside a Harbor trial."""

    def __init__(self, *args, live_token: str = "", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._live_token = live_token

    @staticmethod
    def name() -> str:
        return "openjiuwen-single-harness"

    def version(self) -> str | None:
        return "tb2-harbor"

    async def setup(self, environment: BaseEnvironment) -> None:
        """The agent runs on the host; nothing is installed in the task container."""

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        await live_agent(self._live_token)(instruction, environment)
