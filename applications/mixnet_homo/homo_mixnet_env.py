import functools

import numpy as np
from gymnasium.spaces import Box, Dict

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from applications.mixnet_homo.graph_generator import HOMOGraph


def env():
    env = Mixnet_env()
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class Mixnet_env(AECEnv):
    """
    Note this env is a mixnet with homogeneous nodes and HARD-CODING net params.
    """

    metadata = {"name": "mixnet_homo", "num_players": 2}

    def __init__(self,
                 total_time_step=10,
                 render_mode=None):

        self.possible_agents = ["player_" + str(r) for r in range(2)] # P0: defender, P1: attacker
        self.render_mode = render_mode

        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        self.action_spaces = {}
        self.action_spaces["player_0"] = Box(0.0, 1.0, (6,), np.float32) # 6 = 2 * 3
        self.action_spaces["player_1"] = Box(0.0, 1.0, (6,), np.float32)

        self.observation_spaces = {}
        self.observation_spaces["player_0"] = Box(0.0, 1.0, (6,), np.float32)
        self.observation_spaces["player_1"] = Box(0.0, 1.0, (6,), np.float32)

        self.total_time_steps = total_time_step
        self.graph = HOMOGraph(num_layers=3,
                            num_nodes_per_layer=[1000, 1000, 1000],
                            false_alarm=[0.3, 0.2, 0.3], # false alarm rate
                            false_negative=[0.2, 0.2, 0.2], # false negative rate
                            a_attack_cost=[-400, -500, -400], # attacker's attacking cost
                            a_deploy_cost=[-200, -200, -200], # attacker's deployment cost
                            a_maintain_cost=[-50, -50, -50], # attacker's maintaining cost
                            active_rate=[0.6, 0.3, 0.3], # the possibility of sucessfully activate a node
                            d_exclude_cost=[-300, -300, -300], # defender's cost on excluding a node
                            d_deploy_cost=[-200, -200, -200], # defender's deployment cost
                            d_maintain_cost=[-30, -60, -60],
                            usage_threshold=0.03, # the lower bound of usage without penalty
                            d_penalty=-50, # defender's penalty for insufficient usage
                            a_alpha=60000,  # coefficient for the reward
                            d_beta=50000,
                            normal_nodes=np.array([0.8, 0.8, 0.8]),  # list of ratio of normal nodes
                            compromised_nodes=np.array([0.1, 0.1, 0.1]),
                            seed=None)


    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Box(0.0, 1.0, (6,), np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Box(0.0, 1.0, (6,), np.float32)

    def observe(self, agent):
        return np.array(self.observations[agent])

    def render(self):
        pass

    def close(self):
        pass

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.num_moves = 0
        """
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        """
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        """
        Mixnet
        """
        self.graph.reset()
        self.state = {agent: self.graph.get_graph_state() for agent in self.agents}
        self.observations = {}
        self.observations["player_0"] = self.graph.get_def_observation()
        self.observations["player_1"] = self.graph.get_att_observation()
        self.joint_actions = {agent: None for agent in self.agents}


    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        self._cumulative_rewards[agent] = 0

        # stores action of current agent
        self.joint_actions[self.agent_selection] = action

        # collect reward if it is the last agent to act
        if self._agent_selector.is_last():
            # rewards for all agents are placed in the .rewards dictionary
            if self.joint_actions["player_0"] is None or self.joint_actions["player_1"] is None:
                raise ValueError("Action should not be None.")
            obs_def, obs_att, reward_def, reward_att, state = self.graph.step(np.array(self.joint_actions["player_0"]), np.array(self.joint_actions["player_1"]))

            self.rewards["player_0"], self.rewards["player_1"] = reward_def, reward_att
            self.observations["player_0"], self.observations["player_1"] = obs_def, obs_att
            self.state["player_0"], self.state["player_1"] = state, state

            self.num_moves += 1
            # The truncations dictionary must be updated for all players.
            self.truncations = {
                agent: self.num_moves >= self.total_time_steps for agent in self.agents
            }
            self.joint_actions = {agent: None for agent in self.agents}

        else:
            # no rewards are allocated until both players give an action
            self._clear_rewards()

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()






