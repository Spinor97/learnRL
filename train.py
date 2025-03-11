from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from env import Env

env = Env()
enc = make_vec_env(lambda: env, n_envs=1)

model = PPO("MlpPolicy", enc, verbose=1)

print("Training...")
model.learn(total_timesteps=10000)
print("Training finished.")

print("Saving model...")
model.save("prey_predator")

