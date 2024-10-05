from psro_lib.wrappers.gym_wrapper_recursive import GymPettingZooEnv
from psro_lib.wrappers.gym_wrapper_recursive_reward import GymPettingZooEnv as GymPettingZooEnv_v2

from pettingzoo.classic import rps_v2, tictactoe_v3
from psro_lib.rl_agent_sb3.special_policies_recursive import RandomPolicy_Recursive
from applications.mixnet.mixnet_env import Mixnet_env
from psro_lib.rl_agent_sb3.special_policies_recursive import DeployPolicy_Recursive, FullDefensePolicy_Recursive, ExcludePolicy_Recursive
from psro_lib.rl_agent_sb3.rl_factory import generate_agent_class

import numpy as np

from psro_lib.rl_agent_sb3.rl_oracle import RLOracle, freeze_all

def init_oracle(env):
   # Return the RL policy from SB3. Currently, assume DQN.
   agent_class = generate_agent_class(agent_name="PPO")

   agent_kwargs = {
      "policy": "MlpPolicy",
      "hidden_layers_sizes": 256,
      "hidden_layers": 4,
      "batch_size": 32,
      "learning_rate": 0.001,
   }

   # Create BR oracle.
   oracle = RLOracle(env=env,
                     best_response_class=agent_class,
                     best_response_kwargs=agent_kwargs,
                     total_timesteps=500,
                     sigma=0,
                     verbose=0)

   agents = oracle.generate_new_policies()
   freeze_all(agents)

   # agents = [RandomPolicy(), RandomPolicy()]

   return oracle, agents


petz_env = Mixnet_env()
petz_env.reset()
_, agents = init_oracle(petz_env)

# old_policies = [[RandomPolicy_Recursive(env=petz_env, agent_id=0)], [RandomPolicy_Recursive(env=petz_env, agent_id=1)]]

old_policies = [[agents[0]], [agents[1]]]

meta_probabilities = [[1.0], [1.0]]


env = GymPettingZooEnv_v2(petz_env=petz_env,
                       meta_probabilies=[[1.0], [1.0]],
                       old_policies=old_policies,
                       learning_player_id=1)

observation, info = env.reset()


random_policy = RandomPolicy_Recursive(env=petz_env, agent_id=1)

agent = "player_1"
num_action = petz_env.action_spaces[agent].shape or petz_env.action_spaces[agent].n
num_obs = petz_env.observation_spaces[agent].shape or petz_env.observation_spaces[agent].n


for _ in range(100):
   action, _ = random_policy.predict(observation)
   print("---------------")
   print("action", action)
   observation, reward, termination, truncation, info = env.step(action)

   print("obs:", observation)
   print("rew:", reward)
   print("termination:", termination)
   print("truncation:", truncation)

   if termination or truncation:
      observation, info = env.reset()

