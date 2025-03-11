import gymnasium as gym
from gymnasium import spaces
import numpy as np

class Env(gym.Env):
    def __init__(self):
        super(Env, self).__init__()

        # 0: decrease predating rate, 1: increase predating rate
        self.action_space = spaces.Discrete(2)

        # [prey, predator, resource]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(3,), dtype=np.float32)

        self.state = np.array([150., 30., 500.])

        self.prey_growth_rate = 0.1
        self.predation_rate = 0.01
        self.predator_reproduction_rate = 0.005
        self.predator_death_rate = 0.02
        self.food_consumption_rate = 1.0  
        self.food_regrowth_rate = 5.0 
        self.timestep = 0
        self.timestep = 0

    def step(self, action):
        prey, predator, resource = self.state

        if action == 0:
            predation_rate = max(0, self.predation_rate - 0.001)
        else:
            predation_rate = self.predation_rate + 0.001

        resource -= min(resource, self.food_consumption_rate * prey)
        resource += self.food_regrowth_rate

        prey_change = self.prey_growth_rate * prey * (1 / (np.exp(-resource) + 1)) - predation_rate * prey * predator
        predator_change = self.predator_reproduction_rate * prey * predator - self.predator_death_rate * predator

        prey = max(0, prey + prey_change)
        predator = max(0, predator + predator_change)

        resource = max(0, resource - 0.1 * prey)

        reward = -abs(prey / predator - 5) + 0.1 * resource

        self.state = np.array([prey, predator, resource])
        self.timestep += 1

        done = (self.timestep >= 1000) or (prey == 0) or (predator == 0)
        return self.state, reward, done, {}


    def reset(self):
        self.state = np.array([150., 30., 500.])
        self.timestep = 0
        return self.state
    
