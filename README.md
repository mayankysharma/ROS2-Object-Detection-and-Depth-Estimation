[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

# ROS2-Object-Detection-and-Depth-Estimation
Ros2 wrappers for detection person class using yolov5s and get the depth of the detected class using midas model. Also implement TensorRT optimization on yolov8n model for detection task.

### YoloV5 Depth from detection
Used a yolov5s model to first get human detection and then getting depth map from the midas depth estimation model, a rgb webcam is ued to do the inference.
#### Steps to run the node

Clone the repository
```bash
source /opt/ros/humble/setup.bash
cd ros2_ws/src
git clone https://github.com/mayankysharma/ROS2-Object-Detection-and-Depth-Estimation.git
```
Build
```bash
source /opt/ros/humble/setup.bash
cd ros2_ws/
colcon build
source install/setup.bash
```
Open New terminal to run this node

```bash
source /opt/ros/humble/setup.bash
cd ros2_ws/
source install/setup.bash
ros2 run yolov5_ros2 yolov5_node
```
### TensorRT Optimization

Created a new node that is for optimizing inference time which is done by first exporting the `.pt` model to `.engine` which is done via tensorrt libarary. For this experiment, used a `yolov8n.pt` model as the intial model.

The goal of this optimization is to make detection task suitable for edge device like Nvidia's Jetson Orin Nano. It is to be noted that this node only detects objects, further features may be added in the future.

#### Steps to run this node

Open New terminal to run this node

```bash
source /opt/ros/humble/setup.bash
cd ros2_ws/
source install/setup.bash
ros2 run yolov5_ros2 yolov8_node
```


### Results and Observations

It was found that after TensorRT optimization inference speed increased by atleast `36%` which is like `x2.7` times the speed without optimization. 