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

import pytest

from verl.experimental.agent_loop import nemo_gym_agent_loop


def _trajectory():
    return {
        "input_ids": [10, 11, 12, 13, 14],
        "loss_mask": [0, 0, 1, 0, 1],
        "logprobs": [0.0, 0.0, -0.1, 0.0, -0.2],
        "reward": 0.75,
    }


def test_trajectory_maps_to_native_agent_loop_fields():
    output = nemo_gym_agent_loop.trajectory_to_agent_output(_trajectory())

    assert output.prompt_ids == [10, 11]
    assert output.response_ids == [12, 13, 14]
    assert output.response_mask == [1, 0, 1]
    assert output.response_logprobs == [-0.1, 0.0, -0.2]
    assert output.reward_score == 0.75
    native = output.as_dict()
    assert native["responses"].tolist() == [12, 13, 14]
    assert native["response_mask"].tolist() == [1, 0, 1]
    assert native["rollout_log_probs"].tolist() == pytest.approx([-0.1, 0.0, -0.2])


@pytest.mark.asyncio
async def test_agent_loop_calls_gym_run(monkeypatch):
    captured = {}

    async def fake_post_run(url, request):
        captured.update(url=url, request=request)
        return {"trajectory": _trajectory()}

    monkeypatch.setattr(nemo_gym_agent_loop, "_post_run", fake_post_run)
    loop = object.__new__(nemo_gym_agent_loop.NemoGymAgentLoop)
    loop.gym_run_url = "http://gym:8000/run"
    loop.request_key = "nemo_gym_run_request"

    output = await loop.run(
        {"temperature": 0.7, "top_p": 0.9},
        raw_prompt=[{"role": "user", "content": "solve"}],
        nemo_gym_run_request={"verifier_metadata": {"answer": "42"}},
    )

    assert captured == {
        "url": "http://gym:8000/run",
        "request": {
            "verifier_metadata": {"answer": "42"},
            "responses_create_params": {
                "input": [{"role": "user", "content": "solve"}],
                "temperature": 0.7,
                "top_p": 0.9,
            },
        },
    }
    assert output.response_ids == [12, 13, 14]


def test_trajectory_requires_a_trainable_token():
    with pytest.raises(ValueError, match="no trainable response token"):
        nemo_gym_agent_loop.trajectory_to_agent_output(
            {"input_ids": [1], "loss_mask": [0], "logprobs": [0.0], "reward": 0.0}
        )
