from stable_baselines3 import PPO
from ot2_env_wrapper import OT2Env
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from keras.models import load_model
import keras.backend as K

import cv2

from skimage.morphology import skeletonize

from skan import Skeleton, summarize
from skan.csr import skeleton_to_csgraph

from patchify import patchify, unpatchify

from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops

from simple_pid import PID

from PIL import Image

import time

from task_13_functions import f1, iou, roi_image, padder, predict_all, get_nodes, find_end_node, visualise_landmarks, landmarks_max_coordinates, calculate_root_tip_mm


def calculate_root_tip_mm(landmarks, conversion_factor):
    root_tip_mm_x = (landmarks[1] * conversion_factor)/1000
    root_tip_mm_y = (landmarks[0] * conversion_factor)/1000
    return (root_tip_mm_x, root_tip_mm_y)

num_agents = 1
env = OT2Env(num_agents)

image_path = env.sim.get_plate_image()
image = cv2.imread(image_path, 0)
model = load_model('root_model.h5', custom_objects={'f1':f1, 'iou':iou})

predicted_mask = predict_all(image, model, 256)

image = roi_image(image)
image = padder(image, 256)

image, landmarks = landmarks_max_coordinates(image, predicted_mask)

landmarks_pixels = landmarks

# x, y = zip(*landmarks_pixels)

# plt.figure(figsize=(10, 10))
# plt.imshow(image)
# plt.scatter(x, y, c='red', marker='o', label='Points')
# plt.legend()
# plt.show()

x = 0.10775
y = 0.062
z = 0.2

image_shape = image.shape

print(image_shape)

plate_size_mm = 150
plate_size_pixels = image_shape[0]
conversion_factor = plate_size_mm / plate_size_pixels

landmarks_mm = []

for landmark in landmarks_pixels:

    root_tip_mm = calculate_root_tip_mm(landmark, conversion_factor)

    root_tip_robot_x = root_tip_mm[0] + x
    root_tip_robot_y = root_tip_mm[1] + y
    root_tip_robot_z = z

    landmarks_mm.append((root_tip_robot_x, root_tip_robot_y, root_tip_robot_z))

goal_positions = landmarks_mm

observation, _ = env.reset()

# Load the trained agent
model = PPO.load("model.zip")

for goal_pos in goal_positions:
    # Set the goal position for the robot
    env.goal_position = goal_pos
    # Run the control algorithm until the robot reaches the goal position
    while True:
        action, _states = model.predict(observation, deterministic=True)
        observation, rewards, terminated, truncated, info  = env.step(action)
        # calculate the distance between the pipette and the goal
        distance = observation[3:] - observation[:3] # goal position - pipette position
        # calculate the error between the pipette and the goal
        error = np.linalg.norm(distance)
        print(error, action, goal_pos)
        # Drop the inoculum if the robot is within the required error
        if error < 0.01: # 10mm is used as an example here it is too large for the real use case
            action = np.array([0, 0, 0, 1])
            observation, rewards, terminated, truncated, info  = env.step(action)
            break

        if terminated:
            observation, info = env.reset()