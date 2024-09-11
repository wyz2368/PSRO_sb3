import pyspiel
from shimmy.openspiel_compatibility import OpenSpielCompatibilityV0
from psro_lib.wrappers.gym_wrapper import GymPettingZooEnv


"""
Note: action masking is optional, and can be implemented using either observation or info.

PettingZoo Classic environments store action masks in the observation dict:
mask = observation["action_mask"]
Shimmy’s OpenSpiel environments stores action masks in the info dict:
mask = info["action_mask"]

"""

def run_env_from_spiel(game_name="2048"):
    env = pyspiel.load_game(game_name)
    env = OpenSpielCompatibilityV0(env)

    # print(get_space_sizes(env))

    # env = GymPettingZooEnv(env)
    # print(env.agent_idx)

    print("meta_data:", env.game_name)
    print("possible_agents:", env.possible_agents)
    print("observation_spaces:", env.observation_spaces)
    print("action_spaces:", env.action_spaces)

    # print(type(env))
    # print(isinstance(env, OpenSpielCompatibilityV0))

    # env.reset()
    # for agent in env.agent_iter():
    #     observation, reward, termination, truncation, info = env.last()
    #     print("--------------------")
    #     print("agent:", agent)
    #     print("obs:", observation)
    #     print("rew", reward)
    #     print("term:", termination)
    #     print("trunc:",truncation)
    #     print("info:", info)
    #     if termination or truncation:
    #         action = None
    #     else:
    #         action = env.action_space(agent).sample(info["action_mask"])  # this is where you would insert your policy
    #
    #     print("action:", action)
    #     env.step(action)


    # Test gym wrapper
    gym_env = GymPettingZooEnv(petz_env=env, learning_player_id=0)
    observation, info = gym_env.reset()


    for _ in range(10):
        action = gym_env.action_space.sample(gym_env.action_mask)  # this is where you would insert your policy

        print("---------------")
        print("action", action)
        observation, reward, termination, truncation, info = gym_env.step(action)

        print("obs:", observation)
        print("rew:", reward)
        print("termination:", termination)
        print("truncation:", truncation)

        if termination or truncation:
            break



if __name__ == "__main__":
    # Games:
    # kuhn_poker
    # leduc_poker
    # blotto
    # tic_tac_toe
    run_env_from_spiel("tic_tac_toe")