# NeMo Gym agent loop

The loop reserves one of verl's current policy servers, passes its `/v1` URL
to Gym `/run`, then maps Gym's `{input_ids, loss_mask, logprobs, reward}`
directly to `AgentLoopOutput`. Existing padding, `DataProto`, and PPO code
remain unchanged.

```bash
export NEMO_GYM_URL=http://<gym-host>:12000

python -m verl.trainer.main_ppo \
  actor_rollout_ref.rollout.agent.agent_loop_config_path=examples/nemo_gym/agent_loop.yaml \
  actor_rollout_ref.rollout.agent.default_agent_loop=nemo_gym \
  ...
```

Keep the prompt in the normal verl prompt column and extra Gym `/run` fields
in `nemo_gym_run_request`. The Gym host must be able to reach the rollout
server address advertised by verl. This minimal demo targets vLLM rollouts
and a Gym agent with per-run `policy_base_url` support, such as
`mini_swe_agent_2`; configure Gym's policy model name to match verl's model.

```bash
pytest -q tests/experimental/agent_loop/test_nemo_gym_agent_loop_on_cpu.py
```
