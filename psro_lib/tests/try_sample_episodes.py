"""
Test sample espisoldes in abstract meta trainer.
"""
import numpy as np
from pettingzoo.classic import rps_v2, tictactoe_v3
from psro_lib.rl_agent_sb3.random_policy import RandomPolicy

env = rps_v2.env()
# env = tictactoe_v3.env()
env.reset(seed=42)

def sample_episode(env, policies):
  name_to_id = {}
  for i, player_string in enumerate(env.possible_agents):
    name_to_id[player_string] = i

  rewards = np.zeros(len(env.possible_agents))

  env.reset()
  for agent in env.agent_iter():
    agent_id = name_to_id[agent]
    observation, reward, termination, truncation, info = env.last()
    rewards[agent_id] += reward

    if termination or truncation:
      action = None
    else:
      action, _ = policies[agent_id].predict(observation)

    env.step(action)

  return rewards


policies = [RandomPolicy(env=env, agent_id=0), RandomPolicy(env=env, agent_id=1)]

for _ in range(10):
  rewards = sample_episode(env=env, policies=policies)
  print(rewards)

