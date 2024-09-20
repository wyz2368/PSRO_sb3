# Import the necessary libraries
import gymnasium as gym
from stable_baselines3 import DQN

# Create the environment (CartPole-v1 is a classic control environment)
env = gym.make('CartPole-v1')

# Instantiate the DQN agent
model = DQN('MlpPolicy', env, verbose=1)

# Train the model for a given number of time steps
# model.learn(total_timesteps=10000)


# Evaluate the model
obs, _ = env.reset()
for i in range(1000):
    # Predict the action based on the observation
    action, _states = model.predict(obs)
    print(action)
    # Take the action in the environment
    # obs, reward, done, info = env.step(action)
    # # Render the environment for visualization
    # env.render()
    # if done:
    #     obs = env.reset()

# Close the environment when done
# env.close()
