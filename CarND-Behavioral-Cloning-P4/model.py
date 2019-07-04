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
import pickle

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from random import shuffle
from tqdm import tqdm

# Visualizations will be shown in the notebook.
#%matplotlib inline

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


#===============================================================================================
def preprocess_image(img, top_crop_percent=0.35, bottom_crop_percent=0.15,
                       resize_dim=(64, 64)):
    # Empty Cropping and normalization will be done in cropping and lambda steps in network layer
    # img = crop_image(img, top_crop_percent, bottom_crop_percent)
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



# Test the batch generator
test_num = 2
batch_size=8
for i in range(test_num):
    images, angles = next(generate_batch(samples, batch_size=batch_size))
    print('Batch:', i,', num images: ', len(images), ', num angles:', len(angles))
    # plot_images(images, angles, rows=8, cols=6, figsize=(20,25))
    
# Set image sizes
img_height, img_width, img_channels = images[0].shape
print('Images shape: ', 'HEIGHT: {}'.format(img_height), 'WIDTH: {}'.format(img_width), 'CHANNELS: {}'.format(img_channels)) 


# Split data into training and validation datasets
shuffle(samples)
training_samples, validation_samples = train_test_split(samples, test_size=0.2) 
print("Number of training samples: {}".format(len(training_samples)))
print("Number of validation samples: {}".format(len(validation_samples)))



#===============================================================================================
# Define network
# We implement CNN architecture from the nvidia paper with a few modifications 
# (added elu activations, max-pooling and drop-outs)

from keras.models import Sequential
from keras.layers import Cropping2D
from keras.layers.core import Lambda, Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

# generate the neural network
model = Sequential()


# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(img_height, img_width, img_channels)))

# Crop image to see only a road
model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=(img_height, img_width, img_channels)))

# Layer 1 - Convolution, no of filters- 24, filter size= 5x5, stride= 2x2
model.add(Convolution2D(24, (5,5), strides=(2,2)))
model.add(Activation('elu'))

# Layer 2 - Convolution, no of filters- 36, filter size= 5x5, stride= 2x2
model.add(Convolution2D(36, (5,5), strides=(2,2)))
model.add(Activation('elu'))

# Layer 3 - Convolution, no of filters- 48, filter size= 5x5, stride= 2x2
model.add(Convolution2D(48,(5,5), strides=(2,2)))
model.add(Activation('elu'))

# Layer 4 - Convolution, no of filters- 64, filter size= 3x3, stride= 1x1
model.add(Convolution2D(64, (3,3)))
model.add(Activation('elu'))

# Layer 5 - Convolution, no of filters- 64, filter size= 3x3, stride= 1x1
model.add(Convolution2D(64, (3,3)))
model.add(Activation('elu'))

# Flatten image from 2D to side by side
model.add(Flatten())

# Layer 6 - Fully connected layer 1
model.add(Dense(100))
model.add(Activation('elu'))

# Adding a dropout layer to avoid overfitting.
model.add(Dropout(0.25))

# Layer 7 - fully connected layer 1
model.add(Dense(50))
model.add(Activation('elu'))

# Layer 8 - fully connected layer 1
model.add(Dense(10))
model.add(Activation('elu'))

# Layer 9 - fully connected layer 1
model.add(Dense(1)) #here the final layer will contain one value as this is a regression problem and not classification

# Print a summary of the NN
model.summary()





#===============================================================================================
# checkpoint
checkpoint = ModelCheckpoint("model-{epoch:02d}.h5", monitor='loss', verbose=1, save_best_only=False, mode='max')
callbacks_list = [checkpoint]

#===============================================================================================
# Compile and train the model using the generator function
EPOCHS = 5
BATCH_SIZE = 32

train_generator = generate_batch(training_samples, batch_size=BATCH_SIZE)
valid_generator = generate_batch(validation_samples, batch_size=BATCH_SIZE)

model.compile(loss='mse', optimizer='adam')

# Restore saved models paths
saved_models_paths = glob.glob('model-*.h5')
print(saved_models_paths)
if len(saved_models_paths) > 0:
    print("Restoring latest checkpoint: {} for training".format(saved_models_paths[-1]))
    model = load_model(saved_models_paths[-1])

history = model.fit_generator(train_generator, samples_per_epoch=len(training_samples), 
                    validation_data=valid_generator, nb_val_samples=len(validation_samples), 
                    nb_epoch=EPOCHS)

#===============================================================================================
# Save the model and weights
from keras.models import load_model
model.save("./model.h5")
print("Saved model to disk")

# Save the history
hist_history = history.history
pickle.dump(hist_history, open("hist_history.p", "wb"))