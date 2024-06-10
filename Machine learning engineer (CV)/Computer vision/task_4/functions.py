import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob
import shutil
import tensorflow as tf
from tensorflow import keras
from patchify import patchify, unpatchify
from keras.models import Model
import keras.backend as K
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from keras.callbacks import EarlyStopping

from keras.applications.vgg16 import preprocess_input

def load_data(path, type, combined=False, part='root'):

    file_list = sorted(os.listdir(path))
    data = []

    count = 0
    combined_mask = None
    value = 1

    for filename in file_list:
        if filename.endswith(type):
            
            input_image_path = os.path.join(path, filename)
            input_image = cv2.imread(input_image_path, 0)
            
            if type == '.tif':
                if combined is True:
                    
                    if combined_mask is None:
                        combined_mask = np.zeros_like(input_image)
    
                    combined_mask[input_image == 1] = value 
                    count += 1
                    value += 1
                    
                    if count % 4 == 0:
    
                        data.append(combined_mask)
                        combined_mask = None
                        value = 1

                else:
                    if part == 'root':
                        if part in input_image_path:
                            if 'occluded_root' not in input_image_path:
                                data.append(input_image)
                    elif part == 'seed':
                        if part in input_image_path:
                            data.append(input_image)
                    elif part == 'shoot':
                        if part in input_image_path:
                            data.append(input_image)
                    elif part == 'occluded_root':
                        if part in input_image_path:
                            data.append(input_image)

            else:
                data.append(input_image)
    
    return data

def displaying_images(data, number_of_images=5): 
    
    for idx, img in enumerate(data):
        if idx < number_of_images:
            plt.figure()
            plt.imshow(img, cmap='gray')
            plt.axis('off')  
            plt.show()
        else:
            break

def matching_masks(output_train_folder, output_test_folder, output_no_mask_folder, input_train_folder, input_test_folder, masks_folder, prefixes_to_remove):
    
    os.makedirs(output_train_folder, exist_ok=True)
    os.makedirs(output_test_folder, exist_ok=True)
    os.makedirs(output_no_mask_folder, exist_ok=True)

    # List all images in the train and test folders
    train_images = os.listdir(input_train_folder)
    test_images = os.listdir(input_test_folder)

    # Create a set to keep track of matched images
    matched_images = set()

    # Iterate through the masks and move them to the corresponding train or test folder
    for mask in os.listdir(masks_folder):
        mask_name, mask_extension = os.path.splitext(mask)
        
        # Remove each prefix from the mask name
        for prefix in prefixes_to_remove:
            mask_name = mask_name.replace(prefix, '')
        
        # Initialize lists to store matches for train and test images
        train_matches = [img for img in train_images if f"{mask_name}.png" in img]
        test_matches = [img for img in test_images if f"{mask_name}.png" in img]
        
        # Move the mask to the corresponding train and test folders
        for match in train_matches:
            src_path = os.path.join(masks_folder, mask)
            dest_path = os.path.join(output_train_folder, mask)
            shutil.copy(src_path, dest_path)
            matched_images.add(match)
        
        for match in test_matches:
            src_path = os.path.join(masks_folder, mask)
            dest_path = os.path.join(output_test_folder, mask)
            shutil.copy(src_path, dest_path)
            matched_images.add(match)

        # Print a warning if the modified mask name doesn't match any image (optional)
        if not train_matches and not test_matches:
            print(f"Warning: Mask '{mask_name}' does not match any image.")

    # Move unmatched images to the output_unmatched_folder
    for img in set(train_images + test_images) - matched_images:
        src_path = os.path.join(input_train_folder if img in train_images else input_test_folder, img)
        dest_path = os.path.join(output_no_mask_folder, img)
        shutil.move(src_path, dest_path)

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

    roi = input_im[y:y+side_length, x:x+side_length]
    return roi, y, h, x, w

def preprocess_data(images, masks, patch_size):
    images_list = []
    masks_list = []

    for image, mask in zip(images, masks):
        # Apply ROI to both image and mask

        _, y, h, x, w = roi_image(image)
        
        image_roi = image[y:y+h, x:x+w]
        mask_roi = mask[y:y+h, x:x+w]

        # Perform additional processing as needed
        image_roi = padder(image_roi, patch_size)
        mask_roi = padder(mask_roi, patch_size)

        # Patchify the processed image and mask
        image_patches = patchify(image_roi, (patch_size, patch_size), step=patch_size)
        mask_patches = patchify(mask_roi, (patch_size, patch_size), step=patch_size)

        # Reshape the patches
        image_patches = image_patches.reshape(-1, patch_size, patch_size, 1)
        mask_patches = mask_patches.reshape(-1, patch_size, patch_size, 1)

        images_list.append(image_patches)
        masks_list.append(mask_patches)

    X = np.array(images_list)
    y = np.array(masks_list)

    # Reshape the arrays
    X = X.reshape(-1, patch_size, patch_size, 1)
    y = y.reshape(-1, patch_size, patch_size, 1)

    # Normalize the image data
    X = X / 255.0

    return X, y

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

def simple_unet_model(n_classes, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
# Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = inputs

    # Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
     
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    # Expansive path 
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
     
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
     
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
     
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
     
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', f1, iou])
    model.summary()
    
    return model

def loss_plot(H, title):    
    plt.plot(H.history['loss'], label='loss')
    plt.plot(H.history['val_loss'], label='val_loss')
    plt.title(title)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plot = plt.show()
    return plot

def accuracy_plot(H, title):
    plt.plot(H.history['accuracy'], label='accuracy')
    plt.plot(H.history['val_accuracy'], label='val_accuracy')
    plt.title(title)
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plot = plt.show()
    return plot

def predict_all(images, model, patch_size):

    predictions = []

    for image in images:
    
        image, _, _, _, _ = roi_image(image)
        image = padder(image, patch_size)
        
        patches = patchify(image, (patch_size, patch_size), step=patch_size)
        
        i = patches.shape[0]
        j = patches.shape[1]
        
        patches = patches.reshape(-1, patch_size, patch_size, 1)
        
        preds = model.predict(patches/255)
        
        preds_reshaped = preds.reshape(i, j, patch_size, patch_size)
        
        predicted_mask = unpatchify(preds_reshaped, (i*patch_size, j*patch_size))

        predictions.append(predicted_mask)

    return predictions

