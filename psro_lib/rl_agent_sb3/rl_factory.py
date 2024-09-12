from stable_baselines3 import DQN, PPO, SAC
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy


def generate_agent_class(agent_name:str):
    if agent_name == "DQN":
      return DQN
    elif agent_name == "PPO":
      return PPO
    elif agent_name == "SAC":
      return SAC
    elif agent_name == "MaskablePPO":
      return MaskablePPO
    else:
      raise NotImplementedError("The available oracle classes are DQN, PPO, MaskablePPO, SAC")

def generate_agent_policy(policy_name):
    if policy_name in ["MlpPolicy"]:
        return policy_name
    elif policy_name == "MaskableActorCriticPolicy":
        return MaskableActorCriticPolicy
    else:
        raise NotImplementedError("The available policy classes are MlpPolicy and MaskableActorCriticPolicy")