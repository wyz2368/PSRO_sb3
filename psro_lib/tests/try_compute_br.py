"""
Test RL oracle.
"""
import numpy as np
from pettingzoo.classic import rps_v2, tictactoe_v3
from psro_lib.rl_agent_sb3.random_policy import RandomPolicy
from psro_lib.rl_agent_sb3.rl_factory import generate_agent_class
from shimmy.openspiel_compatibility import OpenSpielCompatibilityV0
from psro_lib.game_factory import get_env_factory
from psro_lib.rl_agent_sb3.special_policies import DeployPolicy, FullDefensePolicy, ExcludePolicy


from psro_lib.rl_agent_sb3.rl_oracle import RLOracle

# env = rps_v2.env()
# env = tictactoe_v3.env()
# env.reset(seed=42)

env = get_env_factory("mixnet_homo")()
env.reset()
#


br_args = {
        "policy": "MlpPolicy",
        "hidden_layers_sizes": 256,
        "hidden_layers": 4}

rl_oracle = RLOracle(env=env,
                     best_response_class=generate_agent_class(agent_name="PPO"),
                     best_response_kwargs=br_args,
                     total_timesteps=100000,
                     sigma=0.0,
                     verbose=0)


# rl_oracle = RLOracle(env=env,
#                      best_response_class=generate_agent_class(agent_name="MaskablePPO"),
#                      best_response_kwargs={"policy": "MaskableActorCriticPolicy"},
#                      total_timesteps=1000,
#                      sigma=0.0,
#                      verbose=1)



# old_policies = [[ExcludePolicy(env=env, agent_id=0)], [ExcludePolicy(env=env, agent_id=1)]]
old_policies = [[FullDefensePolicy(env=env, agent_id=0)], [FullDefensePolicy(env=env, agent_id=1)]]
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
  average_action = {}
  average_action["player_0"] = []
  average_action["player_1"] = []
  for agent in env.agent_iter():
    print("----------------")
    print("Agent:", agent)
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
            action, _ = policies[agent_id][0].predict(observation, action_masks=action_mask)
        elif isinstance(observation, dict):
            action_mask = observation["action_mask"]
            observation = observation["observation"]
            action, _ = policies[agent_id][0].predict(observation, action_masks=action_mask)
        else:
            action, _ = policies[agent_id][0].predict(observation)

    print("Action:", action)
    average_action[agent].append(action)
    env.step(action)

  # print("DEF:", np.mean(average_action["player_0"], axis=0))
  # print("ATT:", np.mean(average_action["player_1"], axis=0))
  return rewards


for _ in range(10):
  rewards = sample_episode(env=env, policies=new_policies)
  print(rewards)