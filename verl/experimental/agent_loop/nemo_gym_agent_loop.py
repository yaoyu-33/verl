# Copyright 2026 NVIDIA Corporation
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

from copy import deepcopy
from typing import Any

import aiohttp

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput


def trajectory_to_agent_output(trajectory: dict[str, Any]) -> AgentLoopOutput:
    """Project Gym's four fields into verl's existing agent-loop contract."""
    input_ids = list(trajectory["input_ids"])
    loss_mask = list(trajectory["loss_mask"])
    logprobs = list(trajectory["logprobs"])
    if not input_ids or len(input_ids) != len(loss_mask) or len(input_ids) != len(logprobs):
        raise ValueError("NeMo Gym trajectory fields must be non-empty and token-aligned")
    try:
        response_start = loss_mask.index(1)
    except ValueError as error:
        raise ValueError("NeMo Gym trajectory has no trainable response token") from error

    return AgentLoopOutput(
        prompt_ids=input_ids[:response_start],
        response_ids=input_ids[response_start:],
        response_mask=loss_mask[response_start:],
        response_logprobs=logprobs[response_start:],
        reward_score=float(trajectory["reward"]),
        num_turns=0,
        metrics=AgentLoopMetrics(),
    )


async def _post_run(url: str, request: dict[str, Any]) -> dict[str, Any]:
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=request) as response:
            response.raise_for_status()
            return await response.json()


class NemoGymAgentLoop(AgentLoopBase):
    """Delegate an entire text trajectory to a NeMo Gym agent server."""

    def __init__(self, *args, gym_url: str, request_key: str = "nemo_gym_run_request", **kwargs):
        super().__init__(*args, **kwargs)
        self.gym_run_url = f"{gym_url.rstrip('/')}/run"
        self.request_key = request_key

    async def run(self, sampling_params: dict[str, Any], priority: int = 0, **kwargs) -> AgentLoopOutput:
        del priority
        request = deepcopy(kwargs.get(self.request_key, {}))
        responses_create_params = request.setdefault("responses_create_params", {})
        responses_create_params["input"] = kwargs["raw_prompt"]
        for source, target in (
            ("temperature", "temperature"),
            ("top_p", "top_p"),
        ):
            if source in sampling_params:
                responses_create_params.setdefault(target, sampling_params[source])

        result = await _post_run(self.gym_run_url, request)
        trajectory = result.get("trajectory")
        if trajectory is None:
            raise ValueError("NeMo Gym /run response did not include trajectory")
        return trajectory_to_agent_output(trajectory)
