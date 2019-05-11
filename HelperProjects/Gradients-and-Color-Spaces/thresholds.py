import numpy as np
import cv2


# Define a function that applies Sobel x or y, 
# then takes an absolute value and applies a threshold.
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    
    # Apply the following steps to img
    
    # 1) Convert to grayscale
    # NOTE!!!:
    # Use cv2.COLOR_RGB2GRAY if you've read in an image using mpimg.imread(). 
    # Use cv2.COLOR_BGR2GRAY if you've read in an image using cv2.imread().
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient == 'y':
        sobel = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    else:
        raise TypeError("Only 'x' and 'y' orientations supported!")
    #print(sobel)
    
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    sobel_binary_output = np.zeros_like(scaled_sobel)
    sobel_binary_output[(scaled_sobel > thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    
    # 6) Return this mask as your binary_output image
    return sobel_binary_output




# Define a function that applies Sobel x and y, 
# then computes the magnitude of the gradient
# and applies a threshold
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    
    # Apply the following steps to img
    
    # 1) Convert to grayscale
    # NOTE!!!:
    # Use cv2.COLOR_RGB2GRAY if you've read in an image using mpimg.imread(). 
    # Use cv2.COLOR_BGR2GRAY if you've read in an image using cv2.imread().
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    #print(sobel)
    
    # 3) Calculate the gradient magnitude (absolute value) 
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    grad_mag = np.uint8(255*grad_mag/np.max(grad_mag))

    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(grad_mag)
    binary_output[(grad_mag >= mag_thresh[0]) & (grad_mag <= mag_thresh[1])] = 1
    
    # 6) Return this mask as your binary_output image
    return binary_output




# Define a function that applies Sobel x and y, 
# then computes the direction of the gradient
# and applies a threshold.
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    # Apply the following steps to img
    
    # 1) Convert to grayscale
    # NOTE!!!:
    # Use cv2.COLOR_RGB2GRAY if you've read in an image using mpimg.imread(). 
    # Use cv2.COLOR_BGR2GRAY if you've read in an image using cv2.imread().
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    grad_dir = np.arctan2(abs_sobely, abs_sobelx)

    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(grad_dir)
    binary_output[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1
    
    # 6) Return this mask as your binary_output image
    return binary_output
