"""
Test RL oracle.
"""
import numpy as np
from pettingzoo.classic import rps_v2, tictactoe_v3
from psro_lib.rl_agent_sb3.random_policy import RandomPolicy
from psro_lib.rl_agent_sb3.rl_factory import generate_agent_class
from shimmy.openspiel_compatibility import OpenSpielCompatibilityV0
from psro_lib.game_factory import get_env_factory
from psro_lib.rl_agent_sb3.special_policies_recursive import DeployPolicy_Recursive, FullDefensePolicy_Recursive, ExcludePolicy_Recursive


from psro_lib.rl_agent_sb3.rl_oracle import RLOracle

# env = rps_v2.env()
# env = tictactoe_v3.env()
# env.reset(seed=42)

env = get_env_factory("mixnet_hetero")()
env.reset()
#

br_args = {
        "policy": "MlpPolicy",
        "hidden_layers_sizes": 256,
        "hidden_layers": 4}

rl_oracle = RLOracle(env=env,
                     best_response_class=generate_agent_class(agent_name="PPO"),
                     best_response_kwargs=br_args,
                     total_timesteps=500000,
                     sigma=0.0,
                     verbose=0)

agents = rl_oracle.generate_new_policies()



# old_policies = [[ExcludePolicy(env=env, agent_id=0)], [ExcludePolicy(env=env, agent_id=1)]]
# old_policies = [[FullDefensePolicy_Recursive(env=env, agent_id=0)], [FullDefensePolicy_Recursive(env=env, agent_id=1)]]
old_policies = [[agents[0]], [agents[1]]]

meta_probabilities = [[1.0], [1.0]]




new_policies = rl_oracle(old_policies=old_policies,
                         meta_probabilities=meta_probabilities,
                         copy_from_prev=False)

# new_policies[0][0] = agents[0]

# new_policies = old_policies

def sample_episode(env, policies):
  name_to_id = {}
  for i, player_string in enumerate(env.possible_agents):
    name_to_id[player_string] = i

  rewards = np.zeros(len(env.possible_agents))

  env.reset()

  for agent in env.agent_iter():
      print("----------------")
      print("Agent:", agent)
      agent_id = name_to_id[agent]
      observation, reward, termination, truncation, info = env.last()
      rewards[agent_id] += reward

      num_action = env.action_spaces[agent].shape or env.action_spaces[agent].n
      num_obs = env.observation_spaces[agent].shape or env.observation_spaces[agent].n
      combinatorial_action = np.concatenate((observation, np.zeros(num_action - 1)))

      # print("a:", policies[0].policy)

      while True:
          if termination or truncation:
              action = None
              break
          else:
              if isinstance(env,
                            OpenSpielCompatibilityV0):  # TODO: This if and elif are not correctly handled for recursive.
                  action_mask = info["action_mask"]
                  observation = observation["observation"]
                  action, _ = policies[agent_id][0].predict(observation, action_masks=action_mask)
              elif isinstance(observation, dict):
                  action_mask = observation["action_mask"]
                  observation = observation["observation"]
                  action, _ = policies[agent_id][0].predict(observation, action_masks=action_mask)
              else:
                  # print("d:", len(combinatorial_action))
                  action, _ = policies[agent_id][0].predict(combinatorial_action)

          print("atomic ACT:", action)
          print("Current Obs:", observation)
          print("Current Comb ACT:", combinatorial_action[int(num_obs[0]/2):])
          if action == num_action - 1 or action == None:
              break
          elif combinatorial_action[int(num_obs[0] / 2) + action] == 1:
              break
          else:
              combinatorial_action[int(num_obs[0] / 2) + action] = 1

      print("ACT:", combinatorial_action[int(num_obs[0]/2):])
      if action is None:
          env.step(action)
      else:
          env.step(combinatorial_action[int(num_obs[0]/2):])

  return rewards


for _ in range(100):
  rewards1 = sample_episode(env=env, policies=old_policies)
  new_policies[1][0] = agents[1]
  rewards2 = sample_episode(env=env, policies=new_policies)
  print(rewards1)
  print(rewards2)