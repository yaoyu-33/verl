# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""NeMo Gym agent loop backed by Gym's token-aligned ``/run`` result."""

import json
from time import perf_counter
from typing import Any
from uuid import uuid4

import aiohttp

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput


class NemoGymAgentLoop(AgentLoopBase):
    """Delegate an entire text trajectory to a NeMo Gym agent server."""

    def __init__(self, *args, gym_url: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.gym_run_url = f"{gym_url.rstrip('/')}/run"

    async def _post_run(self, request: dict[str, Any]) -> dict[str, Any]:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session:
            async with session.post(self.gym_run_url, json=request) as response:
                if response.status >= 400:
                    detail = await response.text()
                    raise RuntimeError(f"NeMo Gym /run returned HTTP {response.status}: {detail}")
                return await response.json()

    async def run(self, sampling_params: dict[str, Any], priority: int = 0, **kwargs) -> AgentLoopOutput:
        del priority
        request = dict(kwargs.get("nemo_gym_run_request", {}))
        responses_create_params = dict(request.get("responses_create_params", {}))
        responses_create_params["input"] = kwargs["raw_prompt"]
        for key in ("temperature", "top_p"):
            if key in sampling_params:
                responses_create_params.setdefault(key, sampling_params[key])
        responses_create_params.setdefault("max_output_tokens", self.rollout_config.response_length)
        metadata = dict(responses_create_params.get("metadata") or {})
        extra_body = metadata.get("extra_body") or {}
        if isinstance(extra_body, str):
            extra_body = json.loads(extra_body)
        metadata["extra_body"] = json.dumps(
            extra_body
            | {
                "logprobs": True,
                "return_token_ids": True,
                "return_tokens_as_token_ids": True,
            }
        )
        responses_create_params["metadata"] = metadata
        responses_create_params["top_logprobs"] = 0
        request["responses_create_params"] = responses_create_params

        policy_address, policy_server = await self.server_manager._acquire_server(uuid4().hex)
        request["policy_base_url"] = f"http://{policy_address}/v1"
        started = perf_counter()
        try:
            global_steps = await policy_server.get_global_steps.remote()
            result = await self._post_run(request)
        finally:
            self.server_manager._release_server(policy_address)
        gym_run_seconds = perf_counter() - started
        trajectory = result["trajectory"]
        response_start = trajectory["loss_mask"].index(1)
        return AgentLoopOutput(
            prompt_ids=trajectory["input_ids"][:response_start],
            response_ids=trajectory["input_ids"][response_start:],
            response_mask=trajectory["loss_mask"][response_start:],
            response_logprobs=trajectory["logprobs"][response_start:],
            reward_score=trajectory["reward"],
            num_turns=0,
            metrics=AgentLoopMetrics(generate_sequences=gym_run_seconds),
            extra_fields={"min_global_steps": global_steps, "max_global_steps": global_steps},
        )
