from stable_baselines3 import DQN, PPO, SAC

def generate_agent_class(agent_name:str):
    if agent_name == "DQN":
      return DQN
    elif agent_name == "PPO":
      return PPO
    elif agent_name == "SAC":
      return SAC
    else:
      raise NotImplementedError("The available oracle classes are DQN, PPO, SAC")

