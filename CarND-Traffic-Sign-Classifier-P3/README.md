## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, you will use what you've learned about deep neural networks and convolutional neural networks to classify traffic signs. You will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, you will then try out your model on images of German traffic signs that you find on the web.

We have included an Ipython notebook that contains further instructions 
and starter code. Be sure to download the [Ipython notebook](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb). 

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting three files: 
* the Ipython notebook with the code
* the code exported as an html file
* a writeup report either as a markdown or pdf file 

Creating a Great Writeup
---
A great writeup should include the [rubric points](https://review.udacity.com/#!/rubrics/481/view) as well as your description of how you addressed each point.  You should include a detailed description of the code used in each step (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Dataset and Repository

1. Download the data set. The classroom has a link to the data set in the "Project Instructions" content. This is a pickled dataset in which we've already resized the images to 32x32. It contains a training, validation and test set.
2. Clone the project, which contains the Ipython notebook and the writeup template.
```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```
---
---
---
---
---

# **Traffic Sign Recognition** 

## Writeup

## Project: Build a Traffic Sign Recognition Classifier

### Solution by luk6xff (June 2019)

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1_0]: ./plots/Random_samples_of_German_traffic_sign_dataset-class-id-0.png "Input data"
[image1_3]: ./plots/Random_samples_of_German_traffic_sign_dataset-class-id-3.png "Input data"
[image1_4]: ./plots/Random_samples_of_German_traffic_sign_dataset-class-id-4.png "Input data"
[image1_8]: ./plots/Random_samples_of_German_traffic_sign_dataset-class-id-8.png "Input data"
[image1_12]: ./plots/Random_samples_of_German_traffic_sign_dataset-class-id-12.png "Input data"
[image1_14]: ./plots/Random_samples_of_German_traffic_sign_dataset-class-id-14.png "Input data"
[image1_24]: ./plots/Random_samples_of_German_traffic_sign_dataset-class-id-24.png "Input data"
[image1_37]: ./plots/Random_samples_of_German_traffic_sign_dataset-class-id-37.png "Input data"
[image1_41]: ./plots/Random_samples_of_German_traffic_sign_dataset-class-id-41.png "Input data"
[image2_0]: ./plots/Histogram%20of%20label%20frequency%20in%20TRAINING%20set.png "Histogram of label frequency in TRAINING set"
[image2_1]: ./plots/Histogram%20of%20label%20frequency%20in%20VALIDATION%20set.png "Histogram of label frequency in VALIDATION set"
[image2_2]: ./plots/Histogram%20of%20label%20frequency%20in%20TEST%20set.png "Histogram of label frequency in TEST set"
[image3_0]: ./plots/Original,%20not%20normalized%20samples%20of%20German%20traffic%20sign%20dataset.png
[image3_1]: ./plots/Normalized%20samples%20of%20German%20traffic%20sign%20dataset.png
[image4_0]: ./plots/A%20few%20augmented,%20normalized%20samples%20of%20German%20traffic%20sign%20dataset.png "A few augmented, normalized samples of German traffic sign dataset"
[image4_1]: ./plots/Histogram%20of%20label%20frequency%20in%20final%20TRAINING%20set.png "Histogram of label frequency in final TRAINING set"
[image5_0]: ./plots/Learning%20Curve%20-%20accuracy.png "Learning Curve - accuracy"
[image5_1]: ./plots/Learning%20Curve%20-%20errors.png "Learning Curve - errors"
[image6]: ./plots/New%20test%20sign%20images%20set.png "New test sign images set"
[image7]: ./plots/Visualization%20of%20softmax%20probabilities%20for%20each%20example.png "Visualization of softmax probabilities for each example"
[image8_0]: ./plots/Random%20image.png "Random image"
[image8_1]: ./plots/Feature%20map%201.png "Feature map 1"
[image8_2]: ./plots/Feature%20map%202.png "Feature map 2"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation. The code and implementation can be found in my [project code](https://github.com/luk6xff/SelfDrivingCarND/blob/master/CarND-Traffic-Sign-Classifier-P3/Traffic_Sign_Classifier.ipynb) 

---

### Data Set Summary & Exploration

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is (32, 32, 3).
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.
Below you can find some random samples of German traffic sign dataset.

![alt text][image1_0]
![alt text][image1_3]
![alt text][image1_4]
![alt text][image1_8]
![alt text][image1_12]
![alt text][image1_14]
![alt text][image1_24]
![alt text][image1_37]
![alt text][image1_41]


Histograms of label frequencies in TRAINING, TEST and VALIDATION sets.
* Histogram of label frequency in TRAINING set.png
![alt text][image2_0]
* Histogram of label frequency in VALIDATION set.png
![alt text][image2_1]
* Histogram of label frequency in TEST set.png
![alt text][image2_2]

### Design and Test a Model Architecture

#### 1. Preprocessing

* As a first step, I decided to convert the images to grayscale because several images in the training were pretty dark and contained only little color und the grayscaling reduces the amount of features and thus reduces execution time. Additionally, several research papers have shown good results with grayscaling of the images.
* I also adjusted a contrast of each image by applying histogram equalization. This is to mitigate the numerous situation in which the image contrast is really poor.
* As a last step, I normalized images from the int values of each pixel [0, 255] to float values with range [0, 1].

Here is an example of a traffic sign images before and after preprocessing.

![alt text][image3_0]
![alt text][image3_1]


I decided to generate additional data because to have a more robust model. I augmented the training dataset. By this I also rebalanced the number of examples for each class_id to 6000 to eliminate biases in the final model.

To add more data to the the data set, I used the following techniques on training dataset:
* image rotation;
* image translation;
* brightness change;


Here is an example of a few augmented images and final histogram of training labels.

![alt text][image4_0]
![alt text][image4_1]



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 grayscale image   							|
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 2x2 stride, valid padding, outputs 10x10x16    |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Fully connected		| input 400, output 120        									|
| RELU					|												|
| Dropout				| 70% keep        									|
| Fully connected		| input 120, output 84        									|
| RELU					|												|
| Dropout				| 70% keep        									|
| Fully connected		| input 84, output 43        									|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located from the 12th to 16th cell of the ipython notebook.

To train the model, I used an LeNet for the most part that was learnt during lessons. 
I used the AdamOptimizer with a learning rate of 0.0005.  
The epochs used was 50 while the batch size was 128. 
Keep probalbility of the dropout layer was 0.7.
Variables were initialized using the truncated normal distribution with mu = 0.0 and sigma = 0.1.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.916
* validation set accuracy of 0.945
* test set accuracy of 0.913

My approach is based on tips from previous lessons concerning convolutional neural nets and deep learning. I started with a modified Lenet-5 implementation that I developed for a previous lab. This actually worked, but gave a validation accuracy of about 0.91. I then added dropout layers after relu of both fully connected layers. It gave me validation accuracy = 94.1%. I think that some of the images are just too dark to see the sign, so they act as noises in the training data and drop out layer can reduce the negative effects on learning. It looks like a very good reqularisation technique to robustify a layer to overfitting.
After this point, most of my time was spent on tuning the hyper-parameters. Specifically, number of epochs and batch size. Batch size was also set by trial and error.
Adding more augmented data gave finally validation accuracy = 94.5% after 50 epochs with batch size=128.

Here are the learning curves (learning accuracy and error)
![alt text][image5_0]
![alt text][image5_1]

### Test a Model on New Images

#### 1.  My new german traffic signs

Here are six German traffic signs that I found on the web:

![alt text][image6]
The first image might be difficult to classify because ...

#### 2. Performance on New Images

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1. Speed limit (30km/h)      		| Speed limit (30km/h)   									| 
| 2. Bumpy road     			| Bumpy road 										|
| 3. Ahead only					| Ahead only											|
| 4. No vehicles	      		| Speed limit (80km/h)					 				|
| 5. Go straight or left			| Go straight or left    							|
| 6. General caution	      		| General caution					 				|


The model was able to correctly guess 5 of the 8 traffic signs, which gives an model accuracy of 83%. This compares favorably to the accuracy on the test set of 91.3%.

#### 3. Model Certainty - Softmax Probabilities

The code for making predictions on my final model is located in the 21th cell of the Ipython notebook.

A list of top_k = 3 probabilities for each image:

* Top 3 model predictions for IMAGE:0 (Target is 01 [Speed limit (30km/h)])  
   Prediction = 01 [Speed limit (30km/h)] with certainty 0.59  
   Prediction = 40 [Roundabout mandatory] with certainty 0.24  
   Prediction = 02 [Speed limit (50km/h)] with certainty 0.06  

* Top 3 model predictions for IMAGE:1 (Target is 22 [Bumpy road])  
   Prediction = 22 [Bumpy road] with certainty 1.00  
   Prediction = 26 [Traffic signals] with certainty 0.00  
   Prediction = 29 [Bicycles crossing] with certainty 0.00  

* Top 3 model predictions for IMAGE:2 (Target is 35 [Ahead only])  
   Prediction = 35 [Ahead only] with certainty 1.00  
   Prediction = 34 [Turn left ahead] with certainty 0.00  
   Prediction = 37 [Go straight or left] with certainty 0.00  

* Top 3 model predictions for IMAGE:3 (Target is 15 [No vehicles])  
   Prediction = 05 [Speed limit (80km/h)] with certainty 0.47  
   Prediction = 07 [Speed limit (100km/h)] with certainty 0.45  
   Prediction = 02 [Speed limit (50km/h)] with certainty 0.08  

* Top 3 model predictions for IMAGE:4 (Target is 37 [Go straight or left])  
   Prediction = 37 [Go straight or left] with certainty 0.95  
   Prediction = 40 [Roundabout mandatory] with certainty 0.02  
   Prediction = 33 [Turn right ahead] with certainty 0.01  

* Top 3 model predictions for IMAGE:5 (Target is 18 [General caution])  
   Prediction = 18 [General caution] with certainty 1.00  
   Prediction = 27 [Pedestrians] with certainty 0.00  
   Prediction = 26 [Traffic signals] with certainty 0.00  


A visualization chart of softmax probabilities:
![alt text][image7]


As you can observe the biggest problem was to recognize `No vehicles sign`. The third prediction for it was about 8% and it was applied for `Speed limit (50km/h)` which is not even close to the correct sign. I think that the reason of this problem is the noise we can observe on the white part of the sign after preprocessing which might be similar to numbers on speed limit signs. 

### Visualization of feature Maps
#### **Random image**
![alt text][image8_0]

#### **Convolution Layer 1 feature maps**
![alt text][image8_1]
As seen from the visualization, the first convolutional layer picks up the basic shapes such as edges from the image. We can also observe that colors are also extracted in this layer.

#### **Convolution Layer 2 feature maps**
![alt text][image8_2]
As seen from the output of the second convolutional layer which seems to picking up from on the output of first layer and trying to recognize pixel blobs (traingle shapes in whole)



### Summary
As it was my first contact with machine learning I am quite excited to have learnt all these aspects how to train CNN to recognize data.
In the near future I will definitely go back to this project and try use different net structures to improve my accuracy.

