"""
Test RL oracle.
"""
import numpy as np
from pettingzoo.classic import rps_v2, tictactoe_v3
from psro_lib.rl_agent_sb3.random_policy import RandomPolicy
from psro_lib.rl_agent_sb3.rl_factory import generate_agent_class


from psro_lib.rl_agent_sb3.rl_oracle import RLOracle

# env = rps_v2.env()
env = tictactoe_v3.env()
env.reset(seed=42)

rl_oracle = RLOracle(env=env,
                     best_response_class=generate_agent_class(agent_name="DQN"),
                     best_response_kwargs={},
                     total_timesteps=1000,
                     sigma=0.0,
                     verbose=1)

old_policies = [[RandomPolicy(env=env, agent_id=0)], [RandomPolicy(env=env, agent_id=0)]]
meta_probabilities = [[1.0], [1.0]]


new_policies = rl_oracle(old_policies=old_policies,
                         meta_probabilities=meta_probabilities,
                         copy_from_prev=False)