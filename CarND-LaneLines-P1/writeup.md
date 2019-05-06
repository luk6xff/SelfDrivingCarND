# **Finding Lane Lines on the Road** 

** Solution by luk6xff (April 2019)
** [My solution notebook](P1-SOLUTION.ipynb)

## Writeup 


---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[input_image]: ./pipeline_steps_images/input_image.jpg "input_image"
[gray_image]: ./pipeline_steps_images/gray.jpg "gray"
[blur_image]: ./pipeline_steps_images/blur.jpg "blur"
[edges_image]: ./pipeline_steps_images/edges.jpg "edges"
[roi_image]: ./pipeline_steps_images/roi.jpg "roi"
[masked_edges_image]: ./pipeline_steps_images/masked_edges.jpg "masked_edges"
[lanes_image]: ./pipeline_steps_images/lanes_image.jpg "lanes_image"
[masked_lanes_image]: ./pipeline_steps_images/masked_lanes_img.jpg "masked_lanes_img"
[final_image]: ./pipeline_steps_images/final_img.jpg "final_img"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consists of a few steps. The main function is ```line_detection_process_image(img)```.
The function takes one argument which is RGB image array provided by ```open_test_image(img_path``` function.

The main steps of ```line_detection_process_image``` function:
1. Converting the RGB input image to grayscale one.
2. Applying Gaussian blur filter on the grayscale image to smooth the noises from the image.
3. Detecting edges on the image by using Canny edge detector.
4. Creating region of interest image (```roi``` where the lines are expected to appear.
5. Creating ```masked_edges_img``` which is just an AND mask of edges image and created roi.
6. Extracting straight lines from the masked edges image by using the Probabilistic Hough Line Transform from OpenCV
7. Extracting lanes from the all straight lines returned by hough transform. This is done in the functon: ```extract_lanes```
   - I choose lines with slope (in deegrees) between (-60,-30) for left_lane and (30,60) for right_lane.
   - I compute a median of intercept point and slope for these lines to get a lane approximation points.
   - The function returns a list of left and right lane endpoints in ```lane_lines``` object.
8. Drawing ```lane_lines``` on ```lanes_image```.
9. Creating ```masked_lanes_img``` which is just an and operation between edges image and created roi.
10. Creating ```final_img``` by image blending operation using OpenCV: ```addWeighted``` function.



Below you can see all the steps being done by ```line_detection_process_image``` function:

Input image:
![alt text][input_image]

Grayscale image:
![alt text][gray_image]

Blur image:
![alt text][blur_image]

Edges image:
![alt text][edges_image]

Roi image:
![alt text][roi_image]

Masked edges and roi image:
![alt text][masked_edges_image]

Lanes image:
![alt text][lanes_image]

Masked lanes and roi image:
![alt text][masked_lanes_image]

Final image:
![alt text][final_image]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when the lines are not easily visible on the image, they cannot be simple extracted or the created ROI is not proper defined for given image, the result lanes might not exactly match lanes from the road.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to define a more accurate ROI for given images, streams.
I would also try to play with lines' slope filter to extract only those which seem to be a real road lanes.
