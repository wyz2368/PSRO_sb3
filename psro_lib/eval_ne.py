from psro_lib.utils import sample_strategy_marginal as strategy_sampler
import numpy as np
from psro_lib import meta_strategies
from psro_lib.utils import init_logger
from shimmy.openspiel_compatibility import OpenSpielCompatibilityV0

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
      print("------------")
      print("player:", agent_id)
      print("action:", action)
      print("state:", env.observe("player_1"))

      env.step(action)

  return rewards

def eval_mixed_strategy(env, policies, meta_probabilies, num_episodes):
    for i in range(num_episodes):
        print("========= {} =========".format(i))
        sampled_policies = strategy_sampler(policies, meta_probabilies)
        sample_episode(env=env, policies=sampled_policies)

