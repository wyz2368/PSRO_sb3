import gymnasium as gym
import numpy as np

from psro_lib import utils
from shimmy.openspiel_compatibility import OpenSpielCompatibilityV0

class GymPettingZooEnv(gym.Env):

    def __init__(self,
                 petz_env,
                 learning_player_id=None,
                 meta_probabilies=None,
                 old_policies=None):

        self.petz_env = petz_env
        self.agents = self.petz_env.possible_agents
        self.learning_player_id = learning_player_id
        self.learning_player_string = self.agents[learning_player_id]
        self.agent_idx = {}
        for i, agent_id in enumerate(self.agents):
            self.agent_idx[agent_id] = i

        self.meta_probabilies = meta_probabilies
        self.old_policies = old_policies

        self.original_observation_space = self.petz_env.observation_spaces[self.learning_player_string]
        self.action_space = self.petz_env.action_spaces[self.learning_player_string]

        if isinstance(self.original_observation_space, gym.spaces.Dict):
            self.observation_space = self.original_observation_space['observation']
            self.obs_dict_flag = True
        else:
            self.observation_space = self.original_observation_space
            self.obs_dict_flag = False

        if isinstance(self.petz_env, OpenSpielCompatibilityV0):
            self.shimmy = True
        else:
            self.shimmy = False

        # Sample an old strategy profile to best respond to.
        self.strategy_sampler = utils.sample_strategy_marginal
        if old_policies is not None:
            self.sampled_policies = self.strategy_sampler(self.old_policies, self.meta_probabilies)

    def reset(self, seed=None, options=None):
        if self.learning_player_id is None:
            raise ValueError("Have not assigned learning player.")
        self.petz_env.reset()
        if self.old_policies is not None:
            self.sampled_policies = self.strategy_sampler(self.old_policies, self.meta_probabilies)

        self.run_until_the_learning_player()
        observation, _, _, _, info = self.petz_env.last()
        if self.obs_dict_flag: # Assume action_mask is in obs when obs is a dict.
            self.action_mask = observation["action_mask"]
            observation = observation["observation"]
        elif self.shimmy and "action_mask" in info:
            self.action_mask = info["action_mask"]
        else:
            num_action = self.action_space.shape or self.action_space.n
            self.action_mask = np.ones(num_action, dtype=np.int8)

        num_action = self.action_space.shape or self.action_space.n
        self.combinatoral_action = self.process_obs(observation, np.zeros(num_action - 1))

        return self.combinatoral_action, info

    def process_obs(self, observation, actions):
        """
        Add the action part to the observation.
        """
        result = np.concatenate((observation, actions))

        return result

    def step(self, action): #TODO: how to deal with None?
        # The learning player takes an action.
        num_action = self.action_space.shape or self.action_space.n
        num_obs = self.observation_space.shape or self.observation_space.n
        if action < num_action - 1 and self.combinatoral_action[int(num_obs[0]/2) + action] == 1:
            repeated_action = True
        else:
            repeated_action = False
        if action == num_action - 1 or repeated_action:
            # print("*************")
            self.petz_env.step(self.combinatoral_action[int(num_obs[0]/2):])
            self.run_until_the_learning_player()
            observation, reward, termination, truncation, info = self.petz_env.last()
            if self.obs_dict_flag: # Assume action_mask is in obs when obs is a dict.
                self.action_mask = observation["action_mask"]
                observation = observation["observation"]
            elif self.shimmy and "action_mask" in info:
                self.action_mask = info["action_mask"]
            else:
                self.action_mask = np.ones(num_action, dtype=np.int8)

            self.combinatoral_action = self.process_obs(observation, np.zeros(num_action - 1))
            return self.combinatoral_action, reward/10, termination, truncation, info
        else:
            self.combinatoral_action[int(num_obs[0]/2) + action] = 1
            # Reward shaping
            rew = 0
            id_to_node = self.petz_env.graph.id_to_node
            node = id_to_node[action]
            state = node.state
            if self.learning_player_id == 0:
                # if state == -1:
                #     rew += node.d_deploy_cost
                # else:
                #     if state == 1:
                #         rew += node.d_defend_rew
                #     rew += node.d_exclude_cost
                if state == 1:
                    rew += node.d_defend_rew
                return self.combinatoral_action, rew, False, False, {}
            else:
                if self.def_actions[action] == 1:
                    if state == -1:
                        rew += node.a_deploy_cost
                    else:
                        rew += node.a_attack_cost
                else:
                    if state == -1:
                        rew += 50
                        rew += node.a_deploy_cost
                    elif state == 1:
                        rew += 100
                        rew += node.a_attack_cost
                    else:
                        rew += node.a_attack_cost

                return self.combinatoral_action, rew, False, False, {}

    def run_until_the_learning_player(self):
        # The step function changes the current player, so retrive current player again.
        current_agent_string = self.petz_env.agent_selection
        num_action = self.action_space.shape or self.action_space.n
        num_obs = self.observation_space.shape or self.observation_space.n

        # run until the learning agent arrives again.
        while current_agent_string != self.learning_player_string:
            observation, reward, termination, truncation, info = self.petz_env.last()
            if self.obs_dict_flag: # Assume action_mask is in obs when obs is a dict.
                action_mask = observation["action_mask"]
                observation = observation["observation"]
            elif self.shimmy and "action_mask" in info:
                action_mask = info["action_mask"]
            else:
                action_space = self.petz_env.action_spaces[current_agent_string]
                num_action = action_space.shape or self.action_space.n
                action_mask = np.ones(num_action, dtype=np.int8)

            combinatorial_action = self.process_obs(observation, np.zeros(num_action - 1))

            while True:
                if termination or truncation:
                    action = None
                    break
                else:
                    if self.old_policies is not None:
                        current_agent_idx = self.agent_idx[current_agent_string]
                        current_policy = self.sampled_policies[current_agent_idx]
                        if self.obs_dict_flag:
                            action, _ = current_policy.predict(observation, action_masks=action_mask)
                        elif self.shimmy and "action_mask" in info:
                            action, _ = current_policy.predict(observation, action_masks=action_mask)
                        else:
                            action, _ = current_policy.predict(combinatorial_action)
                    else:
                        # If policies are not specified, then randomly sample an action.
                        if self.obs_dict_flag:
                            action = self.petz_env.action_space(current_agent_string).sample(action_mask)
                        elif self.shimmy and "action_mask" in info:
                            action = self.petz_env.action_space(current_agent_string).sample(action_mask)
                        else:
                            action = self.petz_env.action_space(current_agent_string).sample()

                if action == num_action - 1 or action == None:
                    break
                elif combinatorial_action[int(num_obs[0]/2) + action] == 1:
                    break
                else:
                    combinatorial_action[int(num_obs[0]/2) + action] = 1

            if action is None:
                self.petz_env.step(action)
            else:
                if self.learning_player_string == "player_1":
                    self.def_actions = combinatorial_action[int(num_obs[0]/2):]
                self.petz_env.step(combinatorial_action[int(num_obs[0]/2):])
            current_agent_string = self.petz_env.agent_selection


    def action_mask(self):
        return self.action_mask

    def update_learning_player(self, new_learning_player_id):
        self.learning_player_id = new_learning_player_id
        self.learning_player_string = self.agents[new_learning_player_id]

    def update_meta_probabilities(self, new_meta_probabilities):
        self.meta_probabilies = new_meta_probabilities

    def update_old_policies(self, new_old_policies):
        self.old_policies = new_old_policies
