import random

import numpy as np

class FullDefensePolicy_Recursive:
    def __init__(self, env, agent_id):
        self.env = env
        self.player_string = env.possible_agents[agent_id]
        self.actions = iter(list(range(31)))

    def predict(self, obs, action_masks=None): #TODO: why does this cause stopiter error?
        try:
            return next(self.actions), None
        except StopIteration:
            self.actions = iter(list(range(31)))
            return next(self.actions), None


class ExcludePolicy_Recursive:
    def __init__(self, env, agent_id):
        self.env = env
        self.player_string = env.possible_agents[agent_id]

    def predict(self, obs, action_masks=None):
        self.update_state()
        return next(self.actions)

    def update_state(self):
        nodes_states = self.env.observe("player_1")
        existing_nodes = np.where(nodes_states >= 0)[0]
        self.actions = iter(list(range(existing_nodes)))


class DeployPolicy_Recursive:
    def __init__(self, env, agent_id):
        self.env = env
        self.player_string = env.possible_agents[agent_id]

    def predict(self, obs, action_masks=None):
        self.update_state()
        return next(self.actions)

    def update_state(self):
        nodes_states = self.env.observe("player_1")
        existing_nodes = np.where(nodes_states >= -1)[0]
        self.actions = iter(list(range(existing_nodes)))

class RandomPolicy_Recursive:
    def __init__(self, env, agent_id):
        self.env = env
        self.player_string = env.possible_agents[agent_id]
        self.num_action = self.env.num_nodes

    def predict(self, obs, action_masks=None):
        return self.num_action - 1, None
        # return random.choice(range(self.num_action)), None

