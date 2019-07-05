# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains starting files for the Behavioral Cloning Project.

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to clone driving behavior. You will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.

We have provided a simulator where you can steer a car around a track for data collection. You'll use image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting five files: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* a report writeup file (either markdown or pdf)
* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)

This README file describes how to output the video in the "Details About Files In This Directory" section.

Creating a Great Writeup
---
A great writeup should include the [rubric points](https://review.udacity.com/#!/rubrics/432/view) as well as your description of how you addressed each point.  You should include a detailed description of the code used (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The following resources can be found in this github repository:
* drive.py
* video.py
* writeup_template.md

The simulator can be downloaded from the classroom. In the classroom, we have also provided sample data that you can optionally use to help train your model.

## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

#### Why create a video

1. It's been noted the simulator might perform differently based on the hardware. So if your model drives succesfully on your machine it might not on another machine (your reviewer). Saving a video is a solid backup in case this happens.
2. You could slightly alter the code in `drive.py` and/or `video.py` to create a video of what your model sees after the image is processed (may be helpful for debugging).

### Tips
- Please keep in mind that training images are loaded in BGR colorspace using cv2 while drive.py load images in RGB to predict the steering angles.

---
---
---
---
---


# **Behavioral Cloning** 

## Writeup

## Project: Behavioral Cloning Project

### Solution by luk6xff (June 2019)

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/Distribution_of_steering_angles_in_training_dataset.png "Model Visualization"
[image2]: ./examples/Sample_dataset_images.png "Sample dataset"
[image3]: ./examples/preprocessed_images.png "Preprocessed images"
[image4]: ./examples/generated_data.png "Generated image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* model.ipynb python notebook containing the code to create and train the model
* README.md (section Writeup) summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The `model.ipynb` (`model.py`) is just an extracted part of it) file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The CNN is a slightly modified Keras implementation of this [paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)(https://devblogs.nvidia.com/deep-learning-self-driving-cars/) from NVIDIA Corporation.
It is a deep convolution network which works well with supervised image classification / regression problems. As the NVIDIA model is well documented, I was able to focus how to adjust the training images to produce the best result with some adjustments to the model to avoid overfitting and adding non-linearity to improve the prediction.

I have added the following adjustments to the model:
* Lambda layer to normalized input images to avoid saturation and make gradients work better.
* An additional dropout layer to avoid overfitting after the convolution layers.
* ELU for activation function for each layer except for the output layer to introduce non-linearity.

The trained model successfully steered the simulated car around track-1 of the beta-simulator provided as part of this project.

A lambda layer normalises the image such that pixel values are in the range (-1,1). This helps with better numerical conditioning of the optimisation problem. The cropping layer takes the input camera images and crops out areas from the top (60 pixels - sky) and bottom (20 pixels - hood) that are largely irrelevant to training. This also reduces computational load as the images are now smaller.


The final structure of my CNN:
* Image normalization -lambda
* Image cropping (60 pixxels from top, 20 pixels from the bottom)
* Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
* Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
* Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
* Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
* Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
* Drop out (0.25)
* Fully connected: neurons: 100, activation: ELU
* Fully connected: neurons: 50, activation: ELU
* Fully connected: neurons: 10, activation: ELU
* Fully connected: neurons: 1 (output)

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layer in order to reduce overfitting after the fully connected layer with rate of 0.25. 
The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving and recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use the described earlier CNN designed by NVIDIA team.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that I added dropout after fully conected layer.

The final step was to run the simulator to see how well the car was driving around track 1. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I recorded more streams and .

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network described in previous chapter.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

```
_______________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_3 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_3 (Cropping2D)    (None, 80, 320, 3)        0         
_________________________________________________________________
conv2d_11 (Conv2D)           (None, 38, 158, 24)       1824      
_________________________________________________________________
activation_17 (Activation)   (None, 38, 158, 24)       0         
_________________________________________________________________
conv2d_12 (Conv2D)           (None, 17, 77, 36)        21636     
_________________________________________________________________
activation_18 (Activation)   (None, 17, 77, 36)        0         
_________________________________________________________________
conv2d_13 (Conv2D)           (None, 7, 37, 48)         43248     
_________________________________________________________________
activation_19 (Activation)   (None, 7, 37, 48)         0         
_________________________________________________________________
conv2d_14 (Conv2D)           (None, 5, 35, 64)         27712     
_________________________________________________________________
activation_20 (Activation)   (None, 5, 35, 64)         0         
_________________________________________________________________
conv2d_15 (Conv2D)           (None, 3, 33, 64)         36928     
_________________________________________________________________
activation_21 (Activation)   (None, 3, 33, 64)         0         
_________________________________________________________________
flatten_3 (Flatten)          (None, 6336)              0         
_________________________________________________________________
dense_9 (Dense)              (None, 100)               633700    
_________________________________________________________________
activation_22 (Activation)   (None, 100)               0         
_________________________________________________________________
dropout_3 (Dropout)          (None, 100)               0         
_________________________________________________________________
dense_10 (Dense)             (None, 50)                5050      
_________________________________________________________________
activation_23 (Activation)   (None, 50)                0         
_________________________________________________________________
dense_11 (Dense)             (None, 10)                510       
_________________________________________________________________
activation_24 (Activation)   (None, 10)                0         
_________________________________________________________________
dense_12 (Dense)             (None, 1)                 11        
=================================================================
Total params: 770,619
Trainable params: 770,619
Non-trainable params: 0
_________________________________________________________________
```

#### 3. Creation of the Training Set & Training Process
To load data I used OpenCV to load the images, by default the images are read by OpenCV in BGR format but we need to convert to RGB as in drive.py it is processed in RGB format.
I noticed that steering angles sample data has a huge peak around 0 as shown in picture below, which means that the model may have a bias to go straight. If we include the left and right cameras with an steering angle offset, this can fix the problem.
![alt text][image1]  

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:
![alt text][image2]  

Normalization and cropping of the image is done in Network layers (Lambda, Cropping).

Then I repeated this process on track two in order to get more data points.  
To augment the data sat, I also flipped images and angles. In augmenting after flipping multiply the steering angle by a factor of -1 to get the steering angle for the flipped image. For example, here is a set of normal and flipped images:
![alt text][image4]  

I used a generator to generate the data so as to avoid loading all the images in the memory and instead generate it at the run time in batches of 32. Even Augmented images are generated inside the generators.  

After splitting samples data, number of training samples: 11740,
and number of validation samples: 2935 (20%).  


I used this training data for training the model. The validation set helped determine if the model was over or under fitting.
Model parameter tuning:
* No of epochs= 5
* Optimizer Used- Adam
* Learning Rate- Default 0.001
* Validation Data split- 0.15
* Generator batch size= 32
* Correction factor- 0.2
* Loss Function Used- MSE(Mean Squared Error as it is efficient for regression problem).

### Outcome

The model can drive on the track 1 without any problem. It does not work well for track 2.

- [Track 1](run_track1.mp4)
- [Track 2](run_track2.mp4)


### References
- NVIDIA model: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
- Udacity Self-Driving Car Simulator: https://github.com/udacity/self-driving-car-sim
