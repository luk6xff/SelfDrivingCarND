# Imports
import pickle
import math
import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import skimage
import csv
import scipy
import copy

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from random import shuffle
from tqdm import tqdm

# Visualizations will be shown in the notebook.
%matplotlib inline

# Some globals
DATA_FOLDER_PATH = './data'
STEERING_COEFFICIENT = 0.2
A4_PORTRAIT = (8.27, 11.69)
A4_LANDSCAPE = A4_PORTRAIT[::-1]


# Read driving_log data
csv_file_path = os.path.join(DATA_FOLDER_PATH,'driving_log.csv')

samples = []
with open(csv_file_path) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        
# Remove the header anymore
samples.pop(0);


def load_image(filename):
    image = cv2.imread(filename)
    return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)


def preprocess_image(img, top_crop_percent=0.35, bottom_crop_percent=0.15,
                       resize_dim=(64, 64)):
    # img = crop_image(img, top_crop_percent, bottom_crop_percent) # Cropping will be done in cropping step in network layer
    # img = resize_image(img, resize_dim)
    return img

def flip_image(image, steering_angle):
    return np.fliplr(image), -1 * steering_angle

def change_brightness(image):
    val = 0.3 + np.random.random()
    return image * val


def augment_image(img, steering_angle):
    img, steering_angle = flip_image(img, steering_angle)
    #img = change_brightness(img)
    return img, steering_angle

# Create data (preprocess, augment)
def create_data(samples, steering_correction):
    images = []
    steering_angles = []
    img_types = ['center', 'left', 'right']
    for sample in samples:
        for i, type_img in enumerate(img_types):
            img = load_image(os.path.join(DATA_FOLDER_PATH, sample[i].strip()))
            # preprocess image
            img = preprocess_image(img)
            angle = float(sample[3])
            if 'center' in sample[i]:
                angle = angle
            elif 'left' in sample[i]:
                angle = angle+steering_correction
            else: #right
                angle = angle-steering_correction
            # Add images to the list
            images.append(img)
            steering_angles.append(angle)
            
            # create augmented image
            img = copy.deepcopy(images[-1])
            steering_angle = steering_angles[-1]
            img, angle = augment_image(img, steering_angle)
            # Load augmented data
            images.append(img)
            steering_angles.append(angle)
            
    return images, steering_angles

# Image generator
def generate_batch(samples, batch_size=32):
    """
    Generator function to avoid loading all images and angles into memory
    """
    num_samples = len(samples)
    while True:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            steering_angles = []
            images, steering_angles = create_data(batch_samples, STEERING_COEFFICIENT)
            X_train = np.array(images)
            y_train = np.array(steering_angles)
            yield (X_train, y_train)






# Split data into training and validation datasets
shuffle(samples)
training_samples, validation_samples = train_test_split(samples, test_size=0.2) 
print("Number of training samples: {}".format(len(training_samples)))
print("Number of validation samples: {}".format(len(validation_samples)))






# Define network
# We implement CNN architecture from the nvidia paper with a few modifications 
# (added relu activations, max-pooling and drop-outs)

from keras.models import Sequential
from keras.layers import Cropping2D
from keras.layers.core import Lambda, Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# generate the neural network
model = Sequential()


# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(img_height, img_width, img_channels)))
#model.add(Lambda(lambda x: x/127.5 - 1.))

# Crop image to see only a road
model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=(img_height, img_width, img_channels)))

# # Layer 1- Convolution, number of filters- 24, filter size= 5x5, stride= 2x2
# model.add(Convolution2D(24, 5, 5, activation=None, subsample=(1,1)))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# # Layer 2- Convolution, no of filters- 36, filter size= 5x5, stride= 2x2
# model.add(Convolution2D(36, 5, 5, activation=None, subsample=(1,1)))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# # Layer 3- Convolution, no of filters- 48, filter size= 5x5, stride= 2x2
# model.add(Convolution2D(48, 5, 5, activation='relu', subsample=(2,2)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Convolution2D(64, 3, 3, activation='relu', subsample=(1,1)))
# model.add(Convolution2D(64, 3, 3, activation='relu', subsample=(1,1)))
# model.add(Flatten())
# model.add(Dropout(0.5))
# model.add(Dense(100))
# model.add(Dense(50))
# model.add(Dense(10))
# model.add(Dense(1))

#layer 1- Convolution, no of filters- 24, filter size= 5x5, stride= 2x2
model.add(Convolution2D(24,5,5,subsample=(2,2)))
model.add(Activation('elu'))

#layer 2- Convolution, no of filters- 36, filter size= 5x5, stride= 2x2
model.add(Convolution2D(36,5,5,subsample=(2,2)))
model.add(Activation('elu'))

#layer 3- Convolution, no of filters- 48, filter size= 5x5, stride= 2x2
model.add(Convolution2D(48,5,5,subsample=(2,2)))
model.add(Activation('elu'))

#layer 4- Convolution, no of filters- 64, filter size= 3x3, stride= 1x1
model.add(Convolution2D(64,3,3))
model.add(Activation('elu'))

#layer 5- Convolution, no of filters- 64, filter size= 3x3, stride= 1x1
model.add(Convolution2D(64,3,3))
model.add(Activation('elu'))

#flatten image from 2D to side by side
model.add(Flatten())

#layer 6- fully connected layer 1
model.add(Dense(100))
model.add(Activation('elu'))

#Adding a dropout layer to avoid overfitting. Here we are have given the dropout rate as 25% after first fully connected layer
model.add(Dropout(0.25))

#layer 7- fully connected layer 1
model.add(Dense(50))
model.add(Activation('elu'))


#layer 8- fully connected layer 1
model.add(Dense(10))
model.add(Activation('elu'))

#layer 9- fully connected layer 1
model.add(Dense(1)) #here the final layer will contain one value as this is a regression problem and not classification

# print a summary of the NN
model.summary()

#===============================================================================================
# compile and train the model using the generator function
EPOCHS = 5
BATCH_SIZE = 32

train_generator = generate_batch(training_samples, batch_size=BATCH_SIZE)
valid_generator = generate_batch(validation_samples, batch_size=BATCH_SIZE)

model.compile(loss='mse', optimizer='adam')
history = model.fit_generator(train_generator, samples_per_epoch=len(training_samples), 
                    validation_data=valid_generator, nb_val_samples=len(validation_samples), 
                    nb_epoch=EPOCHS)

#===============================================================================================
# Save the model and weights
from keras.models import load_model
model.save("./model.h5")
print("Saved model to disk")