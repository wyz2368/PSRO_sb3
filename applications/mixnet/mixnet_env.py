import functools

import numpy as np
from gymnasium.spaces import MultiBinary, MultiDiscrete

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from applications.mixnet.graph_generator_fully_connected import Graph


def env():
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    env = Mixnet_env()
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class Mixnet_env(AECEnv):
    """
    The metadata holds environment constants. From gymnasium, we inherit the "render_modes",
    metadata which specifies which modes can be put into the render() method.
    At least human mode should be supported.
    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {"name": "mixnet", "num_players": 2}

    def __init__(self,
                 num_nodes=100,
                 num_time_steps=20,
                 render_mode=None,
                 instance_path="./instances/net1.pkl"):

        self.possible_agents = ["player_" + str(r) for r in range(2)] # P1: defender, P2: attacker
        self.num_time_steps = num_time_steps
        self.num_nodes = num_nodes
        self.instance_path = instance_path
        self.render_mode = render_mode

        # optional: a mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        #TODO: Change this when we add new actions
        self.action_spaces = {}
        self.action_spaces["player_1"] = MultiBinary(self.num_nodes + 1)
        self.action_spaces["player_2"] = MultiBinary(self.num_nodes + 1)

        self.observation_spaces = {}
        self.observation_spaces["player_1"] = MultiDiscrete([3 for _ in range(self.num_nodes)] + [2 for _ in range(self.num_nodes)])
        self.observation_spaces["player_2"] = MultiDiscrete([3 for _ in range(self.num_nodes)] + [2 for _ in range(self.num_nodes)])

        self.graph = Graph(nodes_per_layer=[25, 25, 25, 25])


    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return MultiDiscrete([3 for _ in range(self.num_nodes)] + [2 for _ in range(self.num_nodes)])

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return MultiBinary(self.num_nodes)

    def render(self):
        pass

    def observe(self, agent):
        return np.array(self.observations[agent])

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
            obs_def, obs_att, reward_def, reward_att, state = self.graph.step(np.array(self.joint_actions["player_0"]),
                                                                              np.array(self.joint_actions["player_1"]))

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