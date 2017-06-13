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

[//]: # (Image References)

[image1]: ./output_images/1-calib.png "Undistorted"
[image2]: ./output_images/2-undistort.png "Road Transformed"
[image3]: ./output_images/threshold_x.png "Binary Example"
[image4]: ./output_images/threshold_color.png "Warp Example"
[image5]: ./examples/straight_lines1.png
[image6]: ./examples/test1.png
[image7]: ./examples/test2.png
[image8]: ./examples/test3.png
[image9]: ./examples/test4.png
[image10]: ./examples/test5.png
[image11]: ./examples/test6.png
[video1]: ./output.mp4 "Video"

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

With the given chessboard images, I put them through opencv functions. The functions and their features are explained below.
- findChessboardCorners: to find chessboard corners in the image.
- drawChessboardCorners: to draw the recognized chessboard lines at the previous step. 
- calibrateCamera: to calculate the calibration values, which are ret, mtx, dist, rvecs, and tvecs.
- undistort: to undistort the original image with the calibration values.

The code is in the first and second cell of mainpipeline.ipynb.
The result shows that the pattern in chessboard is more parellel after undistortion.

![alt text][image1]
fig 1. the original (left), the undistorted (right)


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

From the calibration values I calculated with chessboard images, I undistorted the given test images.

![alt text][image2]
fig 2. the original (left), the undistorted (right)


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combinaton of two thresholding methods.
- thresholding by sobel x-derivative
- thresholding by color channel S in HSV

| Sobel x-derivative        | S channel   | 
|:-------------:|:-------------:| 
| ![alt text][image3]  | ![alt text][image4]      |
table 1.

1) Sobel x-derivative:
  I used Sobel x-derivative because the line we want to recognize is in most case vertically shaped. In other words, the lane's value changes dramatically along the x-axis.

2) S channel:
  Saturation in HSV is effective to recognize the lane. Hue and Value are responsible for color and brightness respectively, and they seem to be not so critical or helpful in this situation, so I opted them out. 


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

For Question 3., 4., 5.:  


|   straight_line1    |   test1     | 
|:-------------:|:-------------:| 
| ![alt text][image5]  | ![alt text][image6]      | 
|   test2    |   test3     | 
| ![alt text][image7]  | ![alt text][image8]      |
table 2.  


My perspective transform function('def perspectivetransform(img, topdown)') is at '2. Undistort, Threshold, and perspective transform' in mainpipeline.ipynb.  

The point for source and destination is calculated like this:

>    offset=[150,0]  
>    corners = np.float32([[190,720],[589,457],[698,457],[1145,720]])
>    src = np.float32([corners[0], corners[1], corners[2], corners[3]])  
>    dst = np.float32([corners[0] + offset, [corners[0,0],0] + offset, [corners[3,0],0] - offset, corners[3] - offset])  

This resulted in the following source and destination points:  

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 589, 457      | 340, 0       | 
| 190, 720      | 340, 720      |
| 1145, 720     | 995, 720      |
| 698, 457      | 995, 0        |

The result birdview image is at bottom left in each cell of the 'table 2'. There is seemingly no abnormality at the result.



#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The lane detection function is under '34. Sliding Window Search' in mainpipeline.ipynb.  
I used the entire function from the course but had to adjust it for introduction of classes and finding fits.  
Basically, 'def find_window_centroids' function scans through an image with convolving windows, and finds the index with maximum convolution value.  
Once the points with maximum convolution value are found, then np.polyfit finds the best fits.  
The result is at the bottom left in each cell of the table 2 and the best fit is indicated as green.  
As you can see, there is anomaly in the result of 'test1.jpg.' I expect it is because the curvature at the moment of taking the picture is quite large. 


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The function for the radius curvature is next to the lane detection function, and also from the course.

In order to find the position with respect to center, I set its center to 640 and multiply the lane position with 3.7/600, which is the ratio of meter to pixel.

> ym_per_pix = 30/720 # meters per pixel in y dimension  
> xm_per_pix = 3.7/700 # meters per pixel in x dimension  
> y_eval = warped.shape[0]  
> left_fit_cr = np.polyfit(np.linspace(0, 719, num=9)\*ym_per_pix, window_centroids[:,0]\*xm_per_pix, 2)  
> right_fit_cr = np.polyfit(np.linspace(0, 719, num=9)\*ym_per_pix, window_centroids[:,1]\*xm_per_pix, 2)  
> left_curverad = ((1 + (2\*left_fit_cr[0]\*y_eval\*ym_per_pix + left_fit_cr[1])\*\*2)\*\*1.5) / np.absolute(2\*left_fit_cr[0])  
> right_curverad = ((1 + (2\*right_fit_cr[0]\*y_eval\*ym_per_pix + right_fit_cr[1])\*\*2)\*\*1.5) / np.absolute(2\*right_fit_cr[0])  
> self.line_pos = self.current_fit[0]\*y_eval\*\*2 +self.current_fit[1]\*y_eval + self.current_fit[2]  
> self.line_base_pos = (self.line_pos - 640)\*3.7/600.0 # 3.7 meters is about 600 pixels in the x direction  


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Referring to tabel 2., the final result, which is in the bottom right at each cell of the table, looks successful.
Also here's other three examples. 

* test4.jpg  
![alt text][image9]  
* test5.jpg  
![alt text][image10]  
* test6.jpg  
![alt text][image11]  
---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to the result video on youtube](https://youtu.be/dHxEyRM8CBM) or [link to the local file](./output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* My pipeline sometimes fails when the curvature is high. To be specific, the displayed curvature on output video is not sanity chechekd.
* To improve it, I need to improve my understanding about curvature, then implement sanity check.
* I feel like I have many 'redundant' functions, especially ones to find best fits. It is because I didn't build it from the scratch but borrowed some of it from the course. 