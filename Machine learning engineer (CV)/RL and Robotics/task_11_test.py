import gymnasium as gym
import stable_baselines3 as sb3
from stable_baselines3 import PPO
from ot2_env_wrapper import OT2Env

env = OT2Env()

model = PPO.load('model.zip')

obs, info = env.reset()

for i in range(1000):

    action = model.predict(obs, deterministic=True)[0]
    obs, reward, terminated, truncated, info = env.step(action)
    print(f'obs: {obs}, reward: {reward}')