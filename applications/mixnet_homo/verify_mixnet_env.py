from applications.mixnet_homo.homo_mixnet_env import Mixnet_env
# from applications.mixnet_homo.random import ContinuousRandomPolicy

from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.data import Collector
from tianshou.env import DummyVectorEnv
from tianshou.policy import RandomPolicy, MultiAgentPolicyManager
from pettingzoo.classic import tictactoe_v3, rps_v2
from pettingzoo.utils import wrappers


def run_env():
    env = Mixnet_env()

    env1 = env
    env1.reset()

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




def run_tsenv():
    env1 = Mixnet_env()
    env1 = wrappers.AssertOutOfBoundsWrapper(env1)
    env1 = wrappers.OrderEnforcingWrapper(env1)
    env = PettingZooEnv(env=env1)
    # env = PettingZooEnv(env=tictactoe_v3.env())
    # env = PettingZooEnv(env=rps_v2.env())
    # rps = rps_v2.env()
    # rps.reset()
    # print(type(rps.observe("player_0")))

    # action = env1.action_space("player_0").sample()
    # print(action)
    # print(env.step(action))

    # obs, reward, done, info = env.step(action)

    # print(obs, reward, done, info)

    # policy = MultiAgentPolicyManager(
    #     policies=[ContinuousRandomPolicy(action_space=env.action_space), ContinuousRandomPolicy(action_space=env.action_space)], env=env
    # )
    policy = MultiAgentPolicyManager(
        policies=[RandomPolicy(action_space=env.action_space),
                  RandomPolicy(action_space=env.action_space)], env=env
    )

    env2 = DummyVectorEnv([lambda: env])

    collector = Collector(policy, env2)
    collector.reset()
    result = collector.collect(n_episode=1)
    print(result)


if __name__ == "__main__":
    # run_env()
    run_tsenv()