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

num_agents = 1
env = OT2Env(num_agents) #Defining the environment

# Taking the random image from the environment
image_path = env.sim.get_plate_image()
image = cv2.imread(image_path, 0)

# Loading the model
model = load_model('root_model.h5', custom_objects={'f1':f1, 'iou':iou})

# predicting the image
predicted_mask = predict_all(image, model, 256)

#extracting the petri dish, then padding it
image = roi_image(image)
image = padder(image, 256)

# Takes the primary root coordinates
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

# From pixels to mm
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

# Taking observations
observation, _ = env.reset()

# Defining gains for the controller
Kp = 15
Ki = 0
Kd = 3

# Creating controller for each axis
pid_controller_x = PID(Kp=Kp, Ki=Ki, Kd=Kd)
pid_controller_y = PID(Kp=Kp, Ki=Ki, Kd=Kd)
pid_controller_z = PID(Kp=Kp, Ki=Ki, Kd=Kd)

error_list = []

# recording the time for benchamrking
start_time_total = time.time()

# iterating through the goal_positions
for goal_pos in goal_positions:

    pid_controller_x.setpoint, pid_controller_y.setpoint, pid_controller_z.setpoint = goal_pos

    while True:
        
        # Taking the current positions
        current_x = observation[0]
        current_y = observation[1]
        current_z = observation[2]

        
        control_output_x = pid_controller_x(current_x)
        control_output_y = pid_controller_y(current_y)
        control_output_z = pid_controller_z(current_z)

        # Giving command to the environment
        observation, reward, terminated, truncated, info = env.step([control_output_x, control_output_y, control_output_z])

        # calculating the error
        error = np.linalg.norm(np.array([control_output_x, control_output_y, control_output_z]))

        if error < 0.001: # Drops water, when under 0.001
            action = np.array([0, 0, 0, 1])
            observation, rewards, terminated, truncated, info  = env.step(action)
            break

        if terminated:
            observation, info = env.reset()
            break

end_time_total = time.time() # stops the recording

print(end_time_total - start_time_total)
print(np.mean(error_list))