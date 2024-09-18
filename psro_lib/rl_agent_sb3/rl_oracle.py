"""
An Oracle for any RL algorithm.
"""
import copy
import torch

from open_spiel.python.algorithms.psro_v2 import optimization_oracle
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import EvalCallback

from psro_lib.wrappers.gym_wrapper import GymPettingZooEnv
from psro_lib.rl_agent_sb3.rl_factory import generate_agent_policy
from sb3_contrib import MaskablePPO

def freeze_all(policies):
  """
  Freezes all policies within policy_per_player.
  """
  for policy in policies:
    for param in policy.policy.parameters():
      param.requires_grad = False

def unfreeze_all(policies):
    """
      Unfreezes all policies within policy_per_player.
    """
    for policy in policies:
      for param in policy.policy.parameters():
        param.requires_grad = True


class RLOracle(optimization_oracle.AbstractOracle):
  """Oracle handling Approximate Best Responses computation."""

  def __init__(self,
               env,
               best_response_class,
               best_response_kwargs,
               total_timesteps,
               sigma=0.0,  # Noise for copying strategies.
               verbose=1,
               **kwargs):
    """
    Init function for the RLOracle.
    """
    self.env = env
    self.gym_env = GymPettingZooEnv(petz_env=env,
                                    learning_player_id=0)


    self.num_players = len(env.agents)

    self._best_response_class = best_response_class
    self._best_response_kwargs = best_response_kwargs

    if self._best_response_kwargs["policy"] == "MaskableActorCriticPolicy":
      def mask_fn(env):
        return env.action_mask
      self.gym_env = ActionMasker(env=self.gym_env, action_mask_fn=mask_fn)

    self.sigma = sigma  # Noise for copying strategies. int(number_training_episodes)
    self.total_timesteps = total_timesteps

    self.verbose = verbose

    super(RLOracle, self).__init__(**kwargs)

  def generate_new_policies(self,
                            copy_from_prev=False,
                            old_policies=None,
                            copy_with_noise=False):
    """
    Generates new policies to be trained into best responses.
    """
    # Create a container of new policies.
    new_policies = []

    # Two ways to create a new policy: copy from the policy from previous iteration or create a new one.
    for player in range(self.num_players):
      # Not copy from previous iterations.
      if not copy_from_prev:
        # Create a DRL policy.
        # TODO: SAC may need action noise. Possibly callbacks.
        policy_arch = [self._best_response_kwargs["hidden_layers_sizes"]] * self._best_response_kwargs["hidden_layers"]
        policy_kwargs = dict(net_arch=dict(pi=policy_arch, vf=policy_arch))
        nn = generate_agent_policy(self._best_response_kwargs["policy"])
        policy = self._best_response_class(policy=nn,
                                           env=self.gym_env,
                                           policy_kwargs=policy_kwargs,
                                           verbose=self.verbose)
      # Copy from previous iterations.
      else:
        if old_policies is None:
          raise ValueError("No previous policy can be duplicated.")
        target_policy = old_policies[player][-1]
        if not isinstance(target_policy, self._best_response_class):
          raise ValueError("The target policy does not belong to the best response class.")
        policy = copy.deepcopy(target_policy)  # Copy the policy from last iteration.
        if copy_with_noise:
          with torch.no_grad():
            for param in policy.policy.parameters():
              param.add_(torch.randn(param.size()) * self.sigma)

      new_policies.append(policy)

    unfreeze_all(new_policies)
    return new_policies


  def train(self, new_policies):
    for learning_player in range(self.num_players):
      self.gym_env.update_learning_player(learning_player)
      self.gym_env.reset()
      current_policy = new_policies[learning_player]
      current_policy.learn(total_timesteps=self.total_timesteps) #TODO: can add more parameters for learning.


  def __call__(self,
               old_policies,
               meta_probabilities,
               copy_from_prev,
               *args,
               **kwargs
               ):

    # Assign strategies and MSS in the empirical game to gym wrapper.
    self.gym_env.update_old_policies(old_policies)
    self.gym_env.update_meta_probabilities(meta_probabilities)

    # Create new policies that are waiting to be trained.
    new_policies = self.generate_new_policies(copy_from_prev=copy_from_prev,
                                                  old_policies=old_policies)
    # Start the training.
    self.train(new_policies=new_policies)

    # Freeze the new policies to keep their weights static.
    freeze_all(new_policies)

    format_policies = []
    for pol in new_policies:
      format_policies.append([pol])
    return format_policies