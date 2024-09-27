import numpy as np

from applications.mixnet.mixnet_env import Mixnet_env


def run_env():
    env = Mixnet_env()

    env.reset()


    # print("states:", env.observe("player_1"))

    i = 1
    for agent in env.agent_iter():
        print("--{}--".format(i))
        print("agent {}".format(agent))
        observation, reward, termination, truncation, info = env.last()
        print("states:", env.observe("player_1"))
        print("observation:", observation)
        print("reward:", reward)
        print("termination:", termination)

        if termination or truncation:
            action = None
        else:
            # this is where you would insert your policy
            # action = env.action_space(agent).sample()
            action = np.ones(30)

        env.step(action)
        print("action:", action)

        i += 1

    # Test reset
    print("==== Test ====")




if __name__ == "__main__":
    run_env()