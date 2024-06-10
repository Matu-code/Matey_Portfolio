from stable_baselines3 import PPO
import gymnasium as gym
import time
import wandb
import os
from wandb.integration.sb3 import WandbCallback
import argparse
from ot2_env_wrapper import OT2Env

# Setting weights and biases - personal tolkin
os.environ['WANDB_API_KEY'] = '1517aeb1f0d1dde6e3c84956fe8b36191c8963b0'

# Initialize Weights and Biases run with project name "task_11" and enable sync with TensorBoard
run = wandb.init(project="task_11", sync_tensorboard=True)

# Set up WandbCallback for model checkpointing and logging
wandb_callback = WandbCallback(model_save_freq=1000,
                                model_save_path=f"model_2/{run.id}",
                                verbose=2,
                                )

# Define the environment using OT2Env
env = OT2Env()

# Set up command-line argument parser for hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.001)  # Learning rate hyperparameter
parser.add_argument("--batch_size", type=int, default=64)          # Batch size hyperparameter
parser.add_argument("--n_steps", type=int, default=2048)           # Number of steps hyperparameter
parser.add_argument("--n_epochs", type=int, default=10)            # Number of epochs hyperparameter

# Parse the command-line arguments
args = parser.parse_args()

# Initialize PPO model with specified hyperparameters
model = PPO('MlpPolicy', env, verbose=1, 
            learning_rate=args.learning_rate, 
            batch_size=args.batch_size, 
            n_steps=args.n_steps, 
            n_epochs=args.n_epochs,
            tensorboard_log=f"runs/{run.id}")

# Set the total number of training timesteps
time_steps = 100000

# Training loop
for i in range(10):
    # Train the model for the specified number of timesteps with WandbCallback
    model.learn(total_timesteps=time_steps, callback=wandb_callback, progress_bar=True, tb_log_name=f"runs/{run.id}")

    # Save the model checkpoints
    model.save(f"Task_11_model_training/{run.id}/{time_steps*(i+1)}")

