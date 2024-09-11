from pettingzoo.classic import rps_v2, tictactoe_v3

# env = rps_v2.env()
env = tictactoe_v3.env()
env.reset(seed=42)

for agent in env.agent_iter():
    print("-----")
    print("agent:", env.agent_selection)
    observation, reward, termination, truncation, info = env.last()

    print("obs:", observation)
    print("rew:", reward)
    print("termination:", termination)
    print("truncation:", truncation)

    if termination or truncation:
        action = None
    else:
        # this is where you would insert your policy
        action = env.action_space(agent).sample()

    print("action", action)

    env.step(action)
    print("agent after step:", env.agent_selection) # step select the next agent.
