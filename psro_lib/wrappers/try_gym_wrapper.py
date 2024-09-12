from psro_lib.wrappers.gym_wrapper import GymPettingZooEnv
from pettingzoo.classic import rps_v2, tictactoe_v3
from psro_lib.rl_agent_sb3.random_policy import RandomPolicy

# env = GymPettingZooEnv(petz_env=rps_v2.env(),
#                        learning_player_id=1)

petz_env = tictactoe_v3.env()
env = GymPettingZooEnv(petz_env=petz_env,
                       learning_player_id=0)

observation, info = env.reset()

print(env.action_space.n)
print(type(observation))

random_policy = RandomPolicy(env=petz_env, agent_id=0)

for _ in range(20):
   print("action_masks:", env.action_mask, type(env.action_mask))
   action, _ = random_policy.predict(observation, action_masks=env.action_mask)
   # action = env.action_space.sample()
   print("---------------")
   print("action", action)
   observation, reward, termination, truncation, info = env.step(action)

   print("obs:", observation)
   print("rew:", reward)
   print("termination:", termination)
   print("truncation:", truncation)

   if termination or truncation:
      observation, info = env.reset()

