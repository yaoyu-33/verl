# NeMo Gym agent loop

Gym `/run` returns `{input_ids, loss_mask, logprobs, reward}`. The connector
maps that directly to verl `AgentLoopOutput`; existing padding, `DataProto`,
and PPO code remain unchanged.

```bash
export NEMO_GYM_URL=http://<gym-host>:12000

python -m verl.trainer.main_ppo \
  actor_rollout_ref.rollout.agent.agent_loop_config_path=examples/nemo_gym/agent_loop.yaml \
  actor_rollout_ref.rollout.agent.default_agent_loop=nemo_gym \
  ...
```

Keep the prompt in the normal verl prompt column and extra Gym `/run` fields
in `nemo_gym_run_request`. Gym's configured model server must route to the
policy being trained.

```bash
pytest -q tests/experimental/agent_loop/test_nemo_gym_agent_loop_on_cpu.py
```
