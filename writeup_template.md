## Advanced Lane Finding - Project Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # "Image References"

[image1]: ./output_images/camera_cal.jpg "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[image7]: ./output_images/test1_undist.jpg "Road Undistorted"
[image8]: ./output_images/straight_lines1_warped.jpg "Road Warped"
[image9]: ./output_images/straight_lines1_threshold.jpg "Road Thresholded"
[image10]: ./output_images/big_warped_img.jpg "Test Images Warped"
[image11]: ./output_images/example_pipeline_out.jpg "Pipeline Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in function `calibrate_camera` located in "./src/calibrate.py".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

For image like this one:
![alt text][image2]

Its undistorted version looks like this:

![alt text][image7]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

After playing with a lot of hyperparameter and methods, the final one that proved to be the most robust is a combination of adaptive thresholding on R channel, combined with simple thresholds on H, S channels of an image transformed to HLS colorspace. Morphological operations (dilation and erosion) were performed on thresholded images in order to get rid of artifacts. Thresholding was performed on an image that is already warped. The code is in `thresholding_pipeline` function, defined in `src/threshold.py`.

Original image:

![alt text][image8]

Thresholded image:

![alt text][image9]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for trasnforming an image is defined in `src/transform.py`, with the main entry function being `transform_road_img`.

The code that defines perspective transform is defined in `get_perspective_transform` function. The function accepts region of interest (ROI), which is expected to contain all lane markings, and offset, which defines left and right margins of warped ROI on the resulting image. The ROI is defined as:

```python
DEFAULT_ROI = [
    # bottom left/right
    (255, 690),
    (1050, 690),
    # top left/right
    (590, 455),
    (695, 455),
]
```

The offset is set to 300. This resulted in the following source and destination points:

|  Source   | Destination |
| :-------: | :---------: |
| 255, 690  |  300, 720   |
| 1050, 690 |  980, 720   |
| 590, 455  |   300, 0    |
| 695, 455  |   980, 0    |

I verified that my perspective transform was working as expected by warping all images from `test_images` directory and making sure all lane markings are visible near the center of an image. The result is visible below:

![alt text][image10]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code for filling the polynomial is copied from relevant lessons, and is defined in `src/line_poly_finder.py`, with the main function being `fit_polynomial`, which accepts warped thresholded image as a parameter.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Again, the code was copied from relevant lessons, the relevant code is in `measure_curvature_pixels` function.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The code in `src/main.py` is responsible for creating processed images that are then assembled into the final video. An example image looks like that:

![alt text][image11]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://youtu.be/Ijz-7MV7Ap4) 

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The most problems occured when road chaged its color to more "bright" one, whhich caused yellow line to be difficult to detect. This shows that it might be difficult to make traditional Computer Vision methods work in every condition imaginable - different pavements might need different hyperparameters to work correctly. One thing to do to make it more robust would be to collect a large dataset of driving videos and train Deep Learning model with task of lane detection. In fact, this is what some commercial applications use for lane detection. One particular example is [an open source autopilot published by Comma.ai](https://github.com/commaai/openpilot).
