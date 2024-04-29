import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation
import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))

class OT2Env(gym.Env):
    def __init__(self, render=False, max_steps=1000):
        super(OT2Env, self).__init__()
        self.render = render
        self.max_steps = max_steps

        # Create the simulation environment
        self.sim = Simulation(num_agents=1)

        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1]), shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.prev_pipette_pos = None

        # Keep track of the number of steps
        self.steps = 0

    def reset(self, seed=None):
        # Being able to set a seed is required for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Reset the state of the environment to an initial state
        # Set a random goal position for the agent, consisting of x, y, and z coordinates within the working area
        self.goal_position = np.random.uniform(low=[-0.187, -0.1705, 0.1201], high=[0.253, 0.2196, 0.2896])
        # Call the environment reset function
        observation = self.sim.reset(num_agents=1)
        # Process the observation and extract relevant information
        pipette_position = np.array(observation[next(iter(observation))]['pipette_position'], dtype=np.float32)
        observation = np.concatenate([pipette_position, np.array(self.goal_position, dtype=np.float32)])

        # Reset the number of steps
        self.steps = 0

        return observation, {}

    def calculate_reward(self, pipette_position):
        # Calculate reward based on the distance to the goal position
        current_distance_to_goal = np.linalg.norm(pipette_position - self.goal_position)
        prev_distance_to_goal = np.linalg.norm(self.prev_pipette_pos - self.goal_position) if self.prev_pipette_pos is not None else current_distance_to_goal

        distance_improvement = prev_distance_to_goal - current_distance_to_goal

        reward = 1.0
        reached_goal_reward = 300
        excessive_steps_penalty = 0.2
        bonus = 0

        # Reward for moving closer to the goal
        if distance_improvement > 0:
            reward += 10 * distance_improvement
        else:
            reward += 0

        # Check if the goal has been reached
        if current_distance_to_goal < 0.01:
            terminated = True
            reward += reached_goal_reward
        else:
            terminated = False

        # Check if the episode should be truncated
        if self.steps >= self.max_steps:
            truncated = True
        else:
            truncated = False

        # Update the previous position for the next step
        self.prev_pipette_pos = pipette_position.copy()

        return reward, terminated, truncated

    def step(self, action):
        # Execute one time step within the environment
        # Since we are only controlling the pipette position, we accept 3 values for the action and need to append 0 for the drop action
        action = np.append(action, 0)

        # Call the environment step function
        observation = self.sim.run([action])

        # Process the observation and extract relevant information
        pipette_position = np.array(observation[next(iter(observation))]['pipette_position'], dtype=np.float32)
        observation = np.concatenate([pipette_position, np.array(self.goal_position, dtype=np.float32)])

        # Calculate the reward
        reward, terminated, truncated = self.calculate_reward(pipette_position)

        # Check if the goal has been reached
        if np.linalg.norm(pipette_position - self.goal_position) < 0.01:
            terminated = True

        # Check if the episode should be truncated
        if self.steps > self.max_steps:
            truncated = True

        info = {}  # We don't need to return any additional information

        # Increment the number of steps
        self.steps += 1

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass

    def close(self):
        self.sim.close()