"""
Test RL oracle.
"""
import numpy as np
from pettingzoo.classic import rps_v2, tictactoe_v3
from psro_lib.rl_agent_sb3.random_policy import RandomPolicy
from psro_lib.rl_agent_sb3.rl_factory import generate_agent_class
from shimmy.openspiel_compatibility import OpenSpielCompatibilityV0


from psro_lib.rl_agent_sb3.rl_oracle import RLOracle

env = rps_v2.env()
# env = tictactoe_v3.env()
env.reset(seed=42)
#
# rl_oracle = RLOracle(env=env,
#                      best_response_class=generate_agent_class(agent_name="DQN"),
#                      best_response_kwargs={"policy": "MlpPolicy"},
#                      total_timesteps=1000,
#                      sigma=0.0,
#                      verbose=1)

#TODO: add assert: non-mask env does not compatiable with maskppo.
rl_oracle = RLOracle(env=env,
                     best_response_class=generate_agent_class(agent_name="MaskablePPO"),
                     best_response_kwargs={"policy": "MaskableActorCriticPolicy"},
                     total_timesteps=1000,
                     sigma=0.0,
                     verbose=1)



old_policies = [[RandomPolicy(env=env, agent_id=0)], [RandomPolicy(env=env, agent_id=1)]]
meta_probabilities = [[1.0], [1.0]]


new_policies = rl_oracle(old_policies=old_policies,
                         meta_probabilities=meta_probabilities,
                         copy_from_prev=False)


def sample_episode(env, policies):
  name_to_id = {}
  for i, player_string in enumerate(env.possible_agents):
    name_to_id[player_string] = i

  rewards = np.zeros(len(env.possible_agents))

  env.reset()
  for agent in env.agent_iter():
    agent_id = name_to_id[agent]
    observation, reward, termination, truncation, info = env.last()
    # print("obs:", observation)
    rewards[agent_id] += reward

    if termination or truncation:
      action = None
    else:
        if isinstance(env, OpenSpielCompatibilityV0):
            action_mask = info["action_mask"]
            observation = observation["observation"]
            action, _ = policies[agent_id].predict(observation, action_masks=action_mask)
        elif isinstance(observation, dict):
            action_mask = observation["action_mask"]
            observation = observation["observation"]
            action, _ = policies[agent_id].predict(observation, action_masks=action_mask)
        else:
            action, _ = policies[agent_id].predict(observation)

    env.step(action)

  return rewards


for _ in range(10):
  rewards = sample_episode(env=env, policies=new_policies)
  print(rewards)