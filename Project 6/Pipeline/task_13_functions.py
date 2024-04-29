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

def f1(y_true, y_pred):
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = TP / (Positives+K.epsilon())
        return recall
    
    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = TP / (Pred_Positives+K.epsilon())
        return precision
    
    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)
    
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def iou(y_true, y_pred):
    def f(y_true, y_pred):
        y_pred = tf.cast(y_pred>0.5, y_pred.dtype)
        intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
        total = K.sum(K.square(y_true),[1,2,3]) + K.sum(K.square(y_pred),[1,2,3])
        union = total - intersection
        return (intersection + K.epsilon()) / (union + K.epsilon())
    return K.mean(f(y_true, y_pred), axis=-1)

def roi_image(input_im, offset=10):

    if len(input_im.shape) == 3 and input_im.shape[2] == 3:
        im_gray = cv2.cvtColor(input_im, cv2.COLOR_BGR2GRAY)
    elif len(input_im.shape) == 2:
        # If the input image is already grayscale, no need to convert
        im_gray = input_im
    else:
        # Handle other cases (e.g., images with more than 3 channels)
        raise ValueError("Unsupported number of channels in input image")

    kernel = np.ones((50, 50), dtype="uint8")

    im_e = cv2.dilate(im_gray, kernel, iterations=1)
    im_closing = cv2.erode(im_e, kernel, iterations=1)

    th, output_im = cv2.threshold(im_closing, 160, 255, cv2.THRESH_BINARY)

    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(output_im)

    area_of_interest = None
    largest_area = 0

    for i in range(1, len(stats)):
        x, y, w, h, area = stats[i]
        if area > largest_area:
            largest_area = area
            area_of_interest = (x, y, w, h)

    x, y, w, h = area_of_interest

    # Adjust width and height to make a perfect square with an offset
    side_length = max(w, h) + 2 * offset
    x = max(0, x - offset)
    y = max(0, y - offset)

    image = cv2.rectangle(input_im, (x, y), (x + side_length, y + side_length), (0, 0, 255), 2)

    roi = input_im[y:y+side_length, x:x+side_length]
    return roi

def padder(image, patch_size):

    h = image.shape[0]
    w = image.shape[1]
    height_padding = ((h // patch_size) + 1) * patch_size - h
    width_padding = ((w // patch_size) + 1) * patch_size - w

    top_padding = int(height_padding/2)
    bottom_padding = height_padding - top_padding

    left_padding = int(width_padding/2)
    right_padding = width_padding - left_padding

    padded_image = cv2.copyMakeBorder(image, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return padded_image

def predict_all(image, model, patch_size):

    image = roi_image(image)
    image = padder(image, patch_size)
        
    patches = patchify(image, (patch_size, patch_size), step=patch_size)
        
    i = patches.shape[0]
    j = patches.shape[1]
        
    patches = patches.reshape(-1, patch_size, patch_size, 1)
        
    preds = model.predict(patches/255)
        
    preds_reshaped = preds.reshape(i, j, patch_size, patch_size)
        
    predicted_mask = unpatchify(preds_reshaped, (i*patch_size, j*patch_size))

    return predicted_mask

def get_nodes(object):

    skeleton = skeletonize(object>0.3)
    summary = summarize(Skeleton(skeleton))
    summary = summary[summary['skeleton-id'] == 0]

    return summary

def find_end_node(graph):
 
    src = list(graph['node-id-src'])
    end_nodes = []
    for destination in list(graph['node-id-dst']):
        if destination not in src:
            end_nodes.append(destination)
    return end_nodes

def visualise_landmarks(object, nodes, show_image=True):
    if show_image:
        plt.imshow(object, cmap='gray')

    start_nodes = list(nodes['node-id-src'])
    end_nodes = find_end_node(nodes)

    min_index = nodes['node-id-src'].idxmin()
    min_value_column1 = nodes.loc[min_index, 'image-coord-src-0']
    min_value_column2 = nodes.loc[min_index, 'image-coord-src-1']

    for node in start_nodes:
        node_info = nodes[nodes['node-id-src'] == node]
        x_start, y_start = node_info['image-coord-src-1'], node_info['image-coord-src-0']

        plt.scatter(x_start, y_start, color='blue')

        # Check if any of the points are equal to the minimum index
        if (x_start == min_value_column2).any():
            plt.scatter(min_value_column2, min_value_column1, color='green')

    max_index = nodes['node-id-dst'].idxmax()
    max_value_column1 = nodes.loc[max_index, 'image-coord-dst-0']
    corresponding_value_column2 = nodes.loc[max_index, 'image-coord-dst-1']

    for node in end_nodes:
        node_info = nodes[nodes['node-id-dst'] == node]
        x_end, y_end = node_info['image-coord-dst-1'], node_info['image-coord-dst-0']

        plt.scatter(x_end, y_end, color='red')

        # Check if any of the points are equal to the maximum index
        if (x_end == corresponding_value_column2).any():
            plt.scatter(corresponding_value_column2, max_value_column1, color='black')

    max_coords = (corresponding_value_column2, max_value_column1)

    if show_image:
        plt.show()

    return max_coords

def landmarks_max_coordinates(image, prediction, num_objects=5):
    def sort_objects_by_area(stats):
        areas = [stat[-1] for stat in stats[1:]]
        sorted_areas = sorted(enumerate(areas, start=1), key=lambda x: x[1], reverse=True)
        return sorted_areas

    def extract_and_highlight_objects(image, prediction, sorted_areas, num_objects):
        plants = []
        plants_pred = []
        all_landmarks = []

        for i in range(1, num_objects + 1):
            index, area = sorted_areas[i - 1]
            x, y, w, h, _ = stats[index]

            cropped_roi_pred = prediction[y:y + h, x:x + w]
            cropped_roi = image[y:y + h, x:x + w]

            nodes = get_nodes(cropped_roi_pred)
            max_coordinates = visualise_landmarks(cropped_roi, nodes, show_image=False)

            x_landmark = max_coordinates[0] + x
            y_landmark = max_coordinates[1] + y

            all_landmarks.append((x_landmark, y_landmark))

        return image, all_landmarks

    pred = cv2.convertScaleAbs(prediction)

    _, _, stats, _ = cv2.connectedComponentsWithStats(pred)

    sorted_areas = sort_objects_by_area(stats)

    num_objects = min(num_objects, len(stats) - 1)

    result_image, all_landmarks = extract_and_highlight_objects(
        image, prediction, sorted_areas, num_objects
    )

    return result_image, all_landmarks


def calculate_root_tip_mm(landmarks, conversion_factor):
    root_tip_mm_x = (landmarks[1] * conversion_factor)/1000
    root_tip_mm_y = (landmarks[0] * conversion_factor)/1000
    return (root_tip_mm_x, root_tip_mm_y)