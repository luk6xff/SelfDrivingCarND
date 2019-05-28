# **Advanced Lane Finding Project** 

** Solution by luk6xff (May 2019)
** [My solution notebook](P2-SOLUTION.ipynb)

## Writeup 


---
**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
---

[//]: # (Image References)

[image0]: ./output_images/distorted_image.png "Distorted"
[image1]: ./output_images/undistorted_image.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./output_images/undistorted_threshlded_image.png "Binary thresholded image"
[image4]: ./output_images/undistorted_with_ROI.png "Perspective transform"
[image5]: ./output_images/undistorted_and_transformed_image.png "Transformed_image"
[image6]: ./output_images/sliding_windows.png "Sliding windows method"
[image7]: ./output_images/polynomial_line.png "Polynomial line (yellow)"
[image8]: ./output_images/back_projected.png "Back projected image"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell called "1. Camera calibration and distortion coefficients" of the IPython notebook solution file located in [P2-SOLUTION.ipynb](P2-SOLUTION.ipynb)

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera intrinsic calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the distorted test image:
![alt text][image0]

using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (implemented in function `apply_threshold`.  Here's an example of my output for this step.
![alt text][image3]


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `top_down_view`.  The `top_down_view` function takes as inputs an image (`img`) and returns warped image and inverse perspective transform matrix.  I chose the hardcode the source and destination points in the following manner:

```python
x_center=640
x_offset_top=420
x_offset_bottom=110
y_top=img_size[1]#720
y_bottom=500

src = np.float32([[x_center+x_offset_bottom,y_bottom],
                  [x_center+x_offset_top,y_top],
                  [x_center-x_offset_top,y_top],
                  [x_center-x_offset_bottom,y_bottom]])

dst = np.float32([[x_center+x_offset_top,0],
                  [x_center+x_offset_top,img_size[1]],
                  [x_center-x_offset_top,img_size[1]],
                  [x_center-x_offset_top,0]])
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image:
![alt text][image4]

and its warped counterpart to verify that the lines appear parallel in the warped image.
![alt text][image5]


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

With no prior information on where the lines are in the image I implemented a method `find_lane_coarse` to extract lane pixels position. I first search for the peaks of the image histogram done along all the columns in the lower half of the image. I expected two peaks in the histogram (left and right lanes). When the peaks have been identified I applied a window around each peak to accumulate all the pixels from the left and right lanes. The rest of the lane pixels is being gathered by sliding the windows upwards and modyfying their position as a function of pixel count within the given window to follow the curvature of the lane. The described method returns coordinates of pixels for two lanes.
![alt text][image6]

Now that we have found all our pixels belonging to each line through the sliding window method, I fit a second order polynomial to the line.

Once I know where the lines are in one frame of video, I can easily do much faster search on subsequent images by searching within the computed polynomial line of the curve as was implemented in method `find_lane_fine`.
![alt text][image7]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

A radius of curvature of the lane has been computed in the following way where x and y are coordinates of pixels that fall on the lane boundary. 

Qe fit a second order polynomial mapping $x = f(y)$ of the form:

$x = Ay^2 + By + C$

The radius of curvature $r_{curve}$ at any point $x$ on the above mapping is given by

$r_{curve} = \frac{[1 + (\frac{dx}{dy})^2]^\frac{3}{2}}{|\frac{d^2x}{dy^2}|}$

where

$\frac{dx}{dy} = 2Ay + B$

and

$\frac{d^2x}{dy^2} = 2A$

The offset to center of the lane is being computed as the difference between the location of the vertical centerline in the image and the middle point of the extracted lane lines.


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The lanes identified in the top view are back-projected into the original undistorted camera image by using the inverse of the projection matrix as was implemented in method `back_projection`
![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The most difficult part of this project was finding and extracting lanes from the images. My piepline is probably not good enough for searching lanes on higly lighted roads, where there is no visible lines on the road etc.
The weakest point of this solution is done by thresholding the input image. I had to find some const thresholds which when poorly chosen can escalate on the quality of the rest of pipeline.
