import numpy as np

class FullDefensePolicy:
    def __init__(self, env, agent_id):
        self.env = env
        self.player_string = env.possible_agents[agent_id]

    def predict(self, obs, action_masks=None):
        if action_masks is not None:
            return np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), None
        else:
            return np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), None

class ExcludePolicy:
    def __init__(self, env, agent_id):
        self.env = env
        self.player_string = env.possible_agents[agent_id]

    def predict(self, obs, action_masks=None):
        if action_masks is not None:
            return np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]), None
        else:
            return np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]), None


class DeployPolicy:
    def __init__(self, env, agent_id):
        self.env = env
        self.player_string = env.possible_agents[agent_id]

    def predict(self, obs, action_masks=None):
        if action_masks is not None:
            return np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]), None
        else:
            return np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]), None