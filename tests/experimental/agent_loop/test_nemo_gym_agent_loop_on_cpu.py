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

from types import SimpleNamespace

import pytest

from verl.experimental.agent_loop import nemo_gym_agent_loop


def _trajectory():
    return {
        "input_ids": [10, 11, 12, 13, 14],
        "loss_mask": [0, 0, 1, 0, 1],
        "logprobs": [0.0, 0.0, -0.1, 0.0, -0.2],
        "reward": 0.75,
    }


@pytest.mark.asyncio
async def test_agent_loop_calls_gym_run(monkeypatch):
    captured = {}

    class FakeRemoteMethod:
        async def remote(self):
            return 7

    class FakePolicyServer:
        get_global_steps = FakeRemoteMethod()

    class FakeServerManager:
        async def _acquire_server(self, request_id):
            captured["request_id"] = request_id
            return "policy:9000", FakePolicyServer()

        def _release_server(self, address):
            captured["released"] = address

    async def fake_post_run(request):
        captured["request"] = request
        return {"trajectory": _trajectory()}

    loop = object.__new__(nemo_gym_agent_loop.NemoGymAgentLoop)
    loop.gym_run_url = "http://gym:8000/run"
    loop.server_manager = FakeServerManager()
    loop.rollout_config = SimpleNamespace(response_length=256)
    monkeypatch.setattr(loop, "_post_run", fake_post_run)

    output = await loop.run(
        {"temperature": 0.7, "top_p": 0.9},
        raw_prompt=[{"role": "user", "content": "solve"}],
        nemo_gym_run_request={"verifier_metadata": {"answer": "42"}},
    )

    assert captured["request"] == {
        "verifier_metadata": {"answer": "42"},
        "policy_base_url": "http://policy:9000/v1",
        "responses_create_params": {
            "input": [{"role": "user", "content": "solve"}],
            "max_output_tokens": 256,
            "metadata": {
                "extra_body": ('{"logprobs": true, "return_token_ids": true, "return_tokens_as_token_ids": true}')
            },
            "temperature": 0.7,
            "top_logprobs": 0,
            "top_p": 0.9,
        },
    }
    assert captured["request_id"]
    assert captured["released"] == "policy:9000"
    assert output.prompt_ids == [10, 11]
    assert output.response_ids == [12, 13, 14]
    assert output.response_mask == [1, 0, 1]
    assert output.response_logprobs == [-0.1, 0.0, -0.2]
    assert output.reward_score == 0.75
    assert output.extra_fields == {"min_global_steps": 7, "max_global_steps": 7}
