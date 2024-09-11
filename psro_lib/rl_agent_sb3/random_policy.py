
class RandomPolicy:
    def __init__(self, env, agent_id):
        self.env = env
        self.player_string = env.possible_agents[agent_id]

    def predict(self, obs, action_masks=None):
        if action_masks is not None:
            return self.env.action_space(self.player_string).sample(action_masks), None
        else:
            return self.env.action_space(self.player_string).sample(), None

