from applications.mixnet_homo.homo_mixnet_env import Mixnet_env


def run_env():
    env = Mixnet_env()

    env1 = env
    env1.reset()

    print(env.metadata["name"])

    i = 1
    for agent in env1.agent_iter():
        print("--{}--".format(i))
        observation, reward, termination, truncation, info = env1.last()
        print("observation:", observation)
        print("reward:", reward)
        print("termination:", termination)

        if termination or truncation:
            action = None
        else:
            # this is where you would insert your policy
            action = env1.action_space(agent).sample()

        env1.step(action)
        print("action:", action)

        i += 1



if __name__ == "__main__":
    run_env()