# NeMo Gym trajectory agent loop

This demo makes NeMo Gym a verl agent loop without adding a second rollout
format. Gym's `/run` endpoint returns:

```text
{input_ids, loss_mask, logprobs, reward}
```

`NemoGymAgentLoop` maps those fields to verl's existing `AgentLoopOutput`:

| NeMo Gym | verl |
| --- | --- |
| leading `input_ids` | `prompt_ids` |
| remaining `input_ids` | `response_ids` |
| response suffix of `loss_mask` | `response_mask` |
| response suffix of `logprobs` | `response_logprobs` |
| `reward` | `reward_score` |

The normal agent-loop postprocessor then creates the padded tensors and
`DataProto`; PPO code is unchanged.

Set the Gym agent URL and select the supplied agent-loop config:

```bash
export NEMO_GYM_URL=http://<gym-host>:12000

python -m verl.trainer.main_ppo \
  actor_rollout_ref.rollout.agent.agent_loop_config_path=examples/nemo_gym/agent_loop.yaml \
  actor_rollout_ref.rollout.agent.default_agent_loop=nemo_gym \
  ...
```

Each dataset row should keep the prompt in verl's configured prompt column and
put any additional Gym `/run` fields in `nemo_gym_run_request`. The connector
sets `responses_create_params.input` from `raw_prompt`, so the request metadata
and the prompt cannot drift apart.

Gym's configured model server must route inference to the policy being trained.
That model boundary is environment configuration, not another trajectory
adapter in verl.

Validate the connector on CPU with:

```bash
pytest -q tests/experimental/agent_loop/test_nemo_gym_agent_loop_on_cpu.py
```
