# Writeup Project2 Advanced Lane Finding Project

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

[//]: # (Image References)

[image1]: ./camera_cal/test_undist.jpg "Undistorted"
[image2]: ./test_images/Transformed.jpg "Road Transformed"
[image3]: ./test_images/binary_combo.jpg "Binary Example"
[image4]: ./test_images/straight_lines1_top_view_JLI.jpg "Warp Example"
[image5]: ./test_images/test6_color_fit_line_JLI.jpg "Fit Visual"
[image6]: ./test_images/lane_line_text_test6_JLI.jpg "Output"
[video1]: ./project_video_color_lane_JLI.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. 

Please consider this document as my Writeup.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

I created a separate file for camera calibration. So that in the future I can use the script to calibration the other cameras.
* The file called `2016-06-29_Camera_Calibration_JLI.ipynb`
* The output is a pickle file: ./camera_cal/PRJ2_JLI_mtx_dist_pickle.p

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The undistort parameters `mtx` and `dist` are found and stored in `PRJ2_JLI_mtx_dist_pickle.p` file, so I can directly use them. The function to undistort the image is:

```
# Step:1
# Undistort the image
def cal_undistort(img,mtx,dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist
```

It in the file `01_Project_code_JLI.ipynb`, `Cell # Step 1:`.
Here is am exmaple image :
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of gradient thresholds (Sobel x) and color (HLS) to generate a binary image ( Cell # Step 2:1 and Cell #  step2:2 in `01_Project_code_JLI.ipynb`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Using the code in file `2019-06-30_Combain_S_sobel.ipynb`, I captured the transfer function `m` and `mivn`. They were saved in a pickle file `JLI_p_t_transfer.p`. For detecting the source (`src`) and destination (`dst`) points, I use a straight line image, apply Canny edge detection and Hough transform tool learned from the previous units to find the gradient of the lane lines in the image. Then, I use these result for the `src` definition. A small tuning also applied to the detected `src` points:

```python
# src
x1 = leftXmin
x2 = leftXmax
x3 = rightXmin
x4 = rightXmax

y1 = int(leftLineK1[0]*x1 + leftLineK1[1])
y2 = int(leftLineK1[0]*x2 + leftLineK1[1])
y3 = int(rightLineK1[0]*x3 + rightLineK1[1])
y4 = int(rightLineK1[0]*x4 + rightLineK1[1])

p1 = [x1,y1]
p2 = [x2+5,y2]
p3 = [x3-2,y3]
p4 = [x4,y4]

# dst
# leave some room for the curve
dx1 = x1 +100
dx4 = x4 -100

dp1 = [dx1,720]
dp2 = [dx1,0]
dp3 = [dx4,0]
dp4 = [dx4,720]

```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 205, 719      | 305, 720      | 
| 580, 461      | 305, 0        |
| 702, 460      | 1006, 0       |
| 1106, 719     | 1006, 720     |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In file `01_Project_code_JLI.ipynb`, `Cell # Step 4` `Cell # Step 5` describes the way I find the starting point and using sliding windows to detect and gather all the data points I need for the 2nd order ployfit. Afterward, my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of the curvature
* The `xm_per_pix` and `ym_per_pix` are the coefficient between pixel distance and real-world length. They are defined in file `2019-06-30_Combain_S_sobel.ipynb`, and stored in `JLI_p_t_transfer.p`. (`Cell # Find the relationship between pixel and distance`)
* I find the real-world radius of the curvature in file `01_Project_code_JLI.ipynb`, `Cell # Step 7`. 
* The distance between lane center and vehicle center is detected using the code, under `Cell # Process` of file `01_Project_code_JLI.ipynb`. (`# step 8: over lay information`)

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

In file `01_Project_code_JLI.ipynb`, I implemented this step in `Cell # Process`, the line `# step 6: back to perspective` and ` # combine original with color lane`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_color_lane_JLI.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

i. I tried the challenge video with the narrow shoulder and road patch. The code finds the road edge of the lane split, instead of the lane lines. 
  * I need to add another layer of image pre-processing to the combined binary image. That filter should be able to filter out the dark spots in the image. I am thinking using RGB threshold to for this.
  
ii. The harder challenge video has excessively brightness change which makes the lane line and the pavement almost same brightness.
  * I talked to my mentor, it looks like the V-channel in the HSV can solve the problem.
  
iii. Vehicle bounds on the road.
  * When the lane moves up and down rapidly, my code has difficulty to find the lines. The solution could be using the previous ployfit curve directly if the lines are out of frame. Instead of finding new points around ployfit line. Or I can reduce the margin of the searching area around the polynomial curve.
  
### Questions
1. How fast can the processors on the vehicle processing their lane line code? 
  * My code takes around 3 to 5 seconds to process one image. (But I am using the Jupytor notebook.)
2. What's the best way of detecting lane lines in the city?
