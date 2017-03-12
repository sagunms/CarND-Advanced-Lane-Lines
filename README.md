# Advanced Lane Finding Project
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)

[birds_eye_view_1]: ./output_images/birds_eye_view_1.png "Warp Example"
[birds_eye_view_2]: ./output_images/birds_eye_view_2.png
[camera_calib_inputs]: ./output_images/camera_calib_inputs.png "Camera calibration inputs"
[compare_straight_lines1]: ./output_images/compare_straight_lines1.png
[compare_straight_lines2]: ./output_images/compare_straight_lines2.png
[compare_test1]: ./output_images/compare_test1.png
[compare_test2]: ./output_images/compare_test2.png
[compare_test3]: ./output_images/compare_test3.png
[compare_test4]: ./output_images/compare_test4.png
[compare_test5]: ./output_images/compare_test5.png
[compare_test6]: ./output_images/compare_test6.png
[histogram_straight_lines1]: ./output_images/histogram_straight_lines1.png
[histogram_straight_lines2]: ./output_images/histogram_straight_lines2.png
[histogram_test1]: ./output_images/histogram_test1.png
[histogram_test2]: ./output_images/histogram_test2.png
[histogram_test3]: ./output_images/histogram_test3.png
[histogram_test4]: ./output_images/histogram_test4.png
[histogram_test5]: ./output_images/histogram_test5.png
[histogram_test6]: ./output_images/histogram_test6.png
[histogram]: ./output_images/histogram.png
[masking_colour]: ./output_images/masking_colour.png
[masking_combined]: ./output_images/masking_combined.png
[masking_sobel]: ./output_images/masking_sobel.png
[straight_lines1]: ./output_images/straight_lines1.jpg
[straight_lines2]: ./output_images/straight_lines2.jpg
[test1]: ./output_images/test1.jpg
[test2]: ./output_images/test2.jpg
[test3]: ./output_images/test3.jpg
[test4]: ./output_images/test4.jpg
[test5]: ./output_images/test5.jpg
[test6]: ./output_images/test6.jpg
[undistort1]: ./output_images/undistort1.png
[undistort2]: ./output_images/undistort2.png
[windowing_and_fit]: ./output_images/windowing_and_fit.png
[yuv_colour]: ./output_images/yuv_colour.png
[video_project]: ./annotated_project_video.mp4 "Project Video"
[video_challenge]: ./annotated_project_video.mp4 "Challenge Video"
[video_challenge2]: ./annotated_project_video.wimp4 "Challenge Video 2"
[overview]: ./output_images/overview.gif "Overview"

Overview
---

Lane finding is one of the important steps for required autonomous driving robots. The algorithm must be robust to changing lighting and weather conditions, curvature and texture of the road. The objective of this project is to identify lane lines using traditional computer vision techniques. Initially camera calibration is done to correct for camera distortion, the video frame is then warped to Bird Eye's view by perspective transformation and this is followed by Colour and Sobel Edge binary masking. The position of the lane lines are estimated by histogram and windowing technique to find base position of the lanes and a second degree polymonial fit computed which form lane lines. Finally, an inverse perspective transformation is applied in the end and the resulting output of the pipleine is an annotated video consisting of highlighted lane line, radius from the centre and offset information.

![alt text][overview]

Project Goals
---

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

Run Instructions
---

The project is written in python and utilises numpy, OpenCV, scikit learn and MoviePy.

Here are the steps required to generate the model from scratch and run the project for vehicle tracking. 

#### Clone my project
```bash
git clone https://github.com/sagunms/CarND-Advanced-Lane-Lines.git
cd CarND-Advanced-Lane-Lines
```

#### Activate conda environment
Follow instructions from [CarND-Term1-Starter-Kit page](https://github.com/udacity/CarND-Term1-Starter-Kit) to setup the conda environment from scratch.
```bash
source activate carnd-term1
```

#### Run vehicle detection project (output video)
```bash
python lanelines.py -i project_video.mp4 -o annotated_project_video.mp4
```

Project Structure
---

### Source Code
The code is divided up into several files which are imported by model.py and main.py.
* `lanelines.py` - Takes input video file, input trained model and outputs annotated video containing highlighted lane lines to the left and right as well as additional information such as vehicle offset from the centre and radius of curvature of centre of the lane in meters.
* `lane_lib/calib.py` - Contains the class `CameraCalibrate` which takes in chessboard images taken at different angles from the same camera and outputs camera matrix and distortion coefficients which is finally saved in `calib.p`. 
* `lane_lib/detector.py` - This is the main module of this project which consists of `LaneLinesDetector` class. This utilises other classes in this project to process an input image from from the video stream and output the final annotated frame consisting of highlighted lane lines and additional information such as offset from centre and radius of curvature from centre. This detection pipeline corrects for camera distortion, warps image into bird's eye view, binary thresholding, computes histogram and uses a window search technique to track the lane lines.
* `lane_lib/masking.py` - Contains `BinaryMasking` class for filtering the input image based on Y and V channels of YUV colour space. In addition, this class also uses Sobel edge detection in x direction and also calculates gradient direction, gradient magnitude.
* `lane_lib/perspective.py` - Consists of `PerspectiveTransform` class which computes the perspective transformation matrix `M` and its inverse `Minv` to warp road images in bird's eye view. This makes it easier to compute the lane curvature which is crucial for this project.
* `lane_lib/debug.py` - Some plotting functions to assist during debugging the project code.

### Miscellaneous Files
* `AdvancedLaneLines.ipynb` - Jupyter notebook for generating various stages of the project to assist during this writeup. Images produced from this notebook can also be found at output_images/*.png
* `Writeup.ipynb` - Jupyter notebook used to construct this writeup. It has the same content as `README.md`.
* `calib.p` - Pickle file containing instrinc camera calibration matrix and distortion coefficient saved as the outcome of `CameraCalibrate` class used during the initialisation of the lane detection pipeline.
* `annotated_project_video.mp4` - The output of the Advanced lane finding project when processing against project_video.mp4 video.
* `annotated_project_video_1.mp4` - The output of the vehicle detection project when processing against challenge_video.mp4 video. 
* `annotated_project_video_2.mp4` - The output of the vehicle detection project when processing against harder_challenge_video.mp4 video. 

Computer Vision Pipeline
---

### Camera Calibration

Here I will briefly discuss how I computed the camera matrix and distortion coefficients. The camera calibration logic is encapsulated in `CameraCalibrate` class in the `lanelines.py` file. This class's constructor takes following arguments.

1. The glob path to the camera images (chessboard pattern) which we are going to use for camera calibration. 
2. The shape of the corners in (X direction, Y direction)
3. An optional calibration pickle filename if available so that the calibration data can simply be read instead of having to compute it from scratch. 

The constructor returns the camera matrix `mtx` and distortion coefficients `dist`. A public method `draw()` is available to draw the chessboard corners. 

The code for this step is also contained in the in the second code cell of the IPython notebook located in `./AdvancedLaneLines.ipynb`.

![alt text][camera_calib_inputs]

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. 

Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `obj_points` and `img_points` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function. The following images show the Distored image, Undistorted Image and the Difference between the two images.

![alt text][undistort1]

### Distortion-correction
The following images demonstrate the distortion correction to a test chessboard image. The difference images helps gauge the amount of correction made to the initial image.

![alt text][undistort2]

### Binary Masking

Correctly identifying lane line pixels is the most important step of this project where the rest of the pipelines rely on. In order to identify lane line, I used a combination of color and gradient thresholds to generate a binary image which is encapsulated in `BinaryMasking` class `(lines 123 - 202)`. This is divided into two major components:

1. Colour binary masking
    - `colour_filter_y_channel` - Color thresholding in Y component of YUV colour space.
    - `colour_filter_v_channel` - Color thresholding in V component of YUV colour space.
2. Sobel binary masking
    - `sobel_gradient_xy` - Sobel operation in X direction
    - `sobel_gradient_magnitude` - Sobel gradient magnitude
    - `sobel_gradient_direction` - Sobel gradient direction

I used YUV colour space for colour binary masking. The Y component determines the brightness of the color (luminance or luma), while the U and V components determine the color itself (the chroma). Here is an example YUV channels the a test RGB image:

![yuv_colour]

Through trial-and-error, threshold values for various masking parameters were hardcoded in the `BinaryMasking` class. These individual masking operations are logically combined into final mask using the following technique in the constructor of same class:

```python
# Combined masking
mask = np.zeros_like(img_d_mag)
mask[((img_masked_y_channel == 1) | (img_masked_v_channel == 1)) & (img_abs_x == 1) | 
      ((img_g_mag == 1) & (img_d_mag == 1))] = 1
```
Here's are examples that illustrate the individual steps of binary masking step.

![masking_colour]

![masking_combined]

![masking_sobel]

### Perspective Transformation (Warp to Bird's Eye View)

Perspective transformation is used to warp the camera image to a Bird's Eye View perspective. This makes it easier to compute the curvature of the lane lines as the both the lines are parallel to each other. Here, I define the source region of interest polygon that constitute the road region within the vanishing point experimentally and the destination rectangle that image should be warped into bird's eye view. Using `cv2.getPerspectiveTransform(src, dst)` OpenCV function, I can compute the perspective transformation matrix `M` and its inverse `Minv`.

The code for my perspective transform is initialised in the class `PerspectiveTransform` int the `lane_lib/perspective.py` file. This computes the transformation matrix `M` and its inverse `Minv` in this constructor.  The constructor takes the image size tuple `img_size` and computes the relative source and destination points dynamically in the following manner:

```python
# Define source image polygonal region of interest
t_roi_y = np.uint(img_size[0] / 1.5)  # top y
b_roi_y = np.uint(img_size[0])        # bottom y

roi_x = np.uint(img_size[1] / 2)
tl_roi_x = roi_x - 0.2 * np.uint(img_size[1] / 2) # top-left x
tr_roi_x = roi_x + 0.2 * np.uint(img_size[1] / 2) # top-right x
bl_roi_x = roi_x - 0.9 * np.uint(img_size[1] / 2) # bottom-left x
br_roi_x = roi_x + 0.9 * np.uint(img_size[1] / 2) # bottom-right x

# Define source image rectangle
src = np.float32([[bl_roi_x, b_roi_y],
                  [br_roi_x, b_roi_y],
                  [tr_roi_x, t_roi_y],
                  [tl_roi_x, t_roi_y]])

# Define destination image rectangle
dst = np.float32([[0, img_size[0]],
                  [img_size[1], img_size[0]],
                  [img_size[1], 0],
                  [0, 0]])
```
This approach was taken instead of hardcoding the coordinates to provide flexibility of resizing the input image. However, for the default image size used in the project video, this resulted in the following source and destination points:

| Source Points | Destination Points |
|:-------------:|:------------------:|
| (64, 720)     | (0, 720)           |
| (1216, 720)   | (1280, 720)        |
| (768, 480)    | (1280, 0)          |
| (512, 480)    | (0, 0)             |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][birds_eye_view_1]

In `LaneLineDetector` class, lines 34-35 in `draw` method uses `cv2.warpPerspective` OpenCV function to do the actual transformation given the `M` matrix computed as explained above.

### Identification and Tracking of Lane Line positions

After binary masking and perspective tranformation steps, next is to identify lane-line pixels and fit their positions with a polynomial. The base lane-line positions are initialised for the first video frame and the lane-line positions of subsequent frames are updated by searching within a predefined margin only which increases the efficiency of the algorithm. 

#### init() method: Compute base positions by computing histogram and windowing and fit lane lines

In `LaneLineDetector` class, lines 40-45 in `draw` method, the `initialised` flag is used to call `init` method first which initialises the base position of left and right lane lines. For this, I computed a histogram of pixel occurances in x-direction and split from the centre position to find the left and right peaks of the histogram. These peaks are stored for use as the initial base position. The following figure shows the histogram computation for the given binary masked image.

![alt text][histogram_test3]

I used vertically stacked windows to divide the image height into eight parts. Then, I identified the x and y positions of all non-zero pixels in the image and stored them separately for left and right lanes. Then, I iterate through the verticle windows one by one and find the window boundaries in x and y covering both lane lines. These boundaries help limit the search to only a certain margin which is the width of each window. The best left and right x pixel positions are found and if minimum number of pixel threshold is satisfied, this position is saved for the next window search. The average of left and right x positions is found and this is repeated for each of the window from bottom to top. 

After we have x positions for each window, I used `np.polyfit` numpy function to fit a second degree polynomial line for each of the lane lines. I have used safety checks in the code to ensure there are no empty pixels in the image after binary thresholding. Otherwise, the frame is rejected. 

#### update() method: Faster search around precomputed base positions to fit lane lines

When the next video frame comes, the `initialised` flag will direct to `update` method instead. This step will salvage the base x positions computed in the `init` method which was  computationally expensive due to histogramming and windowing. It will limit the `polyfit` points search to a small margin around the base positions which will be a lot faster. If this frame does not have any points to fit, it will terminate the update and unset the `initialised` flag so that we recompute the histogram and windows again. 

The following figure illustrates the whole process. It contains the eight search window rectangles on each side, left and right fit polynomial lines, and the highlighted lane area between the lines. It also shows the pixels from binary masking step.

![alt text][windowing_and_fit]


### Calculation of Radius of Curvature of the lane and Position of vehicle from center

This step discusses the calculation of radius of curvature of the lane and the position of the vehicle with respect to center. The lane curvature is calculated in the `lane_curvature` method in `LaneLinesDetector` class which is in the `lane_lib/detector.py` file (lines 75-102). This is called at the end of `draw` method to annotate the final output lane line image with the following information.
* Offset from centre: The offset of the vehicle from the center of the lane in metres.
* Direction of offset: This is measured by the sign of the offset.
* Radius of Curvature: The deviation of the vehicle off from the center of the road lane in metres. This is the average of curvatures of the left and right lane lines. 

### Inverse Perspective Transformation (Unwarp)

I implemented this step in lines 68 in my code in `detector.py` in the method `draw()` of `LaneLinesDetector` class. It uses `cv2.wrapPerspective` OpenCV function and inverse perspective transformation matrix `Minv` computed earlier to plot back down onto the road such that the lane area is highlighted clearly. Here is an example of my result on a test image:

![alt text][compare_test3]

Project Video
---

The success criteria for this project was that wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road. My detection pipeline performs reasonably well on the entire project video. 

![alt text][overview]

* Here is a [link to my main project video result](./annotated_project_video.mp4).
* Here are links to the [challenge video result](./annotated_project_video_1.mp4) and [harder challenge video result](./annotated_project_video_2.mp4) which did not generalise so well as the first one. 
* **Update:** This project has been integrated into [Vehicle Detection and Tracking project](https://github.com/sagunms/CarND-Vehicle-Detection).

Discussion
---

### Issues and limitations

Advanced Lane Lines project took a very a large amount of time compared to other projects related to self-driving car. The hyper-parameter tuning process for Binary Masking in my computer vision pipeline was extremely tedious and time-consuming. 

My pipeline works quite well in the main project video. The challenge video also works fairly well but, except at one instance under the overhead bridge, the lane lines are completely are lost but quickly recovers. My algorithm is unable to generalize across all the different road conditions, especially demonstrated by the harder challenge video which failed miserably. This shows that traditional computer vision is extremely sensitive to the tuned parameters which needs to be chosen very carefully with no guarantee it would work for a slightly different scenario, let alone an unstructured environment. Therefore, it seems is not a good approach for developing the entirety of the computer vision pipleline for self-driving cars.

This project let me to appreciate the modern deep learning approaches even more. Deep learning approach avoid the need for fine-tuning these parameters as it can learn the optimum colour space itself with the training examples given and are inherently more robust. 

### Future Improvments

For extensions and future directions, I would like to highlight following points.

1. I would improve my computer vision pipeline in more detail and instead of meticulously tuning each and every hyperparmeters, I would probably automate this process. One idea I can think of is by iterating between numerous combination of colour spaces, thresholds, gradient magnitude and direction parameters, etc. and after running this for videos in different road conditions, I could gather statistical data with certain criteria to converge on the decision of colour spaces, parameters, and other computer vision operations like morphological operators, etc.
2. Other approaches such as Kalman filters would probably be better for a more robust way of stabalising lane lines when it deviates from the current road curvature. In the future, I would like to implement at least a Linear Kalman filter to improve this project.
3. I would like to explore other machine learning techniques (both traditional and deep learning) to address lane finding problem.
