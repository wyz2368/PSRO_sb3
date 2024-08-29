import functools

import numpy as np
from gymnasium.spaces import Box, Dict

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from applications.mixnet_homo.graph_generator import HOMOGraph


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
    Note this env is a mixnet with homogeneous nodes and HARD-CODING net params.
    """

    metadata = {"name": "mixnet_homo", "num_players": 2}

    def __init__(self,
                 total_time_step=5,
                 render_mode=None):
        """
        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - render_mode

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.
        """
        self.possible_agents = ["player_" + str(r) for r in range(2)] # P0: defender, P1: attacker
        self.render_mode = render_mode

        # optional: a mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        self._action_spaces = {}
        self._action_spaces["player_0"] = Box(0.0, 1.0, (6,), np.float32)
        self._action_spaces["player_1"] = Box(0.0, 1.0, (6,), np.float32)

        # self.observation_spaces = {
        #     i: Dict(
        #         {
        #             "observation": Box(low=0, high=1.0, shape=(6,), dtype=np.float32),
        #             "action_mask": Box(low=0.0, high=1, shape=(9,), dtype=np.float32),
        #         }
        #     )
        #     for i in self.agents
        # }

        self._observation_spaces = {}
        self._observation_spaces["player_0"] = Box(0, 1.0, (6,), np.float32)
        self._observation_spaces["player_1"] = Box(0, 1.0, (6,), np.float32)

        self.total_time_steps = total_time_step
        self.graph = HOMOGraph(num_layers=3,
                            num_nodes_per_layer=[1000, 1000, 1000],
                            false_alarm=[0.3, 0.2, 0.3], # false alarm rate
                            false_negative=[0.2, 0.2, 0.2], # false negative rate
                            a_attack_cost=[-400, -500, -400], # attacker's attacking cost
                            a_deploy_cost=[-200, -200, -200], # attacker's deployment cost
                            a_maintain_cost=[-50, -50, -50], # attacker's maintaining cost
                            active_rate=[0.3, 0.2, 0.3], # the possibility of sucessfully activate a node
                            d_exclude_cost=[-200, -200, -200], # defender's cost on excluding a node
                            d_deploy_cost=[-100, -100, -100], # defender's deployment cost
                            d_maintain_cost=[-30, -30, -30],
                            usage_threshold=0.03, # the lower bound of usage without penalty
                            d_penalty=-50, # defender's penalty for insufficient usage
                            a_alpha=50000,  # coefficient for the reward
                            d_beta=50000,
                            seed=100)


    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return Box(0.0, 1.0, (6,), np.float32)

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Box(0.0, 1.0, (6,), np.float32)

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        pass

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        # observation of one agent is the previous state of the other

        # return {"observation": np.array(self.observations[agent]), "action_mask": np.zeros(6, "int8")}
        return np.array(self.observations[agent])


    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - terminations
        - truncations
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.
        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """
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
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - terminations
        - truncations
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            # handles stepping an agent which is already dead
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next dead agent,  or if there are no more dead agents, to the next live agent
            self._was_dead_step(action)
            return

        agent = self.agent_selection

        # the agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0
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

