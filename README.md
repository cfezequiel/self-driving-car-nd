# Udacity Self-Driving Car Nanodegree

This repository contains solutions to projects in the [Udacity Self-Driving Car Nanodegree course](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013).

## Projects

### [P1 : Finding lane lines using computer vision](./P1_lane_finding)

Image processing pipeline for identifying road lane lines from an image depicting the driver's side view of the road from inside a car. The pipeline used Hough transform to identify lines after using the Canny algorithm for edge detection. The detected Hough lines were filtered based on their slopes. The images were first color thresholded against white/yellow color and grayscaled. An ROI filter was used to mask out pixels not within the region where lane lines are expected. 

<p align="center">
  <img src="./P1_lane_finding/output_images/solidWhiteRight.gif"><br>
  <em>Output of lane finding pipeline on sample video clip.</em>
</p>

### [P2:  Advanced lane finding](./P2_adv_lane_finding)

More advanced image processing pipeline that detects lane lines and computes radius of curvature of the lane and vehicle's offset from center of the lane. The each camera image was first undistored based on computed camera parameters from test checkboard images. Color and gradient thresholding was then applied to the undistorted image, followed by an ROI boundary mask to filter out pixels outside the lane line region. Perspective transform was applied to the resulting binary image to render a "bird's eye view" of the image. Another ROI filter was applied to the persective transformed image.

Lane pixel extraction involved first finding reference bottom x coordinates for the left and right lanes based on a histogram of the perspective transformed binary image. Positions that had the most number of white pixels on the left and right halves of the image were chosen as starting locations for the left and right lanes respectively. A sliding window approach was used to search for pixels on an upward direction. Once candidate lane pixels were identified, 2nd order polynomial curves could be calculated to draw the lane detected lane lines. For succeeding frames in a video, these polynomial curves were used as reference cetner points to search for subsequent lane pixels rather than the sliding window approach, which improved performance.

Once the lane boundary curves have been identified, they were used to compute the radius of curvature and vehicle offset from lane center. An image with a lane boundary is drawn and overlayed on the original undistored image.

<p align="center">
	<img src="./P2_adv_lane_finding/output_project_video.gif"><br>
	<em>Output of advanced lane finding pipeline on sample video clip.</em>
</p>

### [P3: Traffic sign classifier using CNNs](./P3_traffic_sign_classifier)

CNN classifier based on the [LeNet architecture](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) for identifying German traffic signs. The model was trained on  32x32 color images of the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) with train, validation and test splits of 34799, 4410 and 12630 samples respectively. Model achieved an overal test set accuracy of 94.7% on 43 different traffic sign categories.

<p align="center">
	<img src="./P3_traffic_sign_classifier/writeup_images/sample38_c1.png"><br>
	<em>Feature map outputted by first convolution layer for a "keep right" traffic sign image.</em>
</p>

### [P4: Behavioral cloning for autonomous driving](./P4_behavioral_cloning)

Behavioral cloning (regression) model based on the [NVIDIA paper (Bojarski et al. 2016)](https://arxiv.org/abs/1604.07316). Model was trained on recordings of manually-driven car around simulated tracks using the [Udacity driving simulator](https://github.com/udacity/self-driving-car-sim). Data preprocessing and augmentation, such as brightness changes, random shadow overlay and horizontal flip, was performed to expand the quantity and variety of images for training. 

<p align="center">
	<img src="./P4_behavioral_cloning/writeup_images/final_model_track1_screencast.gif">
	<center>
    <em>Screencast of autonomous driving on track 1 simulation. <a href="https://www.youtube.com/watch?v=N1Pnjn8Hze4">(YouTube video)</a></em>
  </center>
</p>

### [P5: Sensor fusion with extended kalman filter](./P5_extended_kalman_filter)

Extended kalman filter for tracking the position of a moving vehicle in simulation using LIDAR and radar data. 
Implemented in C++.

<p align="center">
	<img src="./P5_extended_kalman_filter/ekf_dataset1.gif"><br>
	<em>Screencast EKF simulation on dataset1.</em>
</p>




## License
[Apache 2.0](./LICENSE)



