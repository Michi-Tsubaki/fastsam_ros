# fastsam_ros

This package provides a ROS1 wrapper for the [Fast Segment Anything Model (FastSAM)](https://github.com/CASIA-IVA-Lab/FastSAM), allowing for real-time segmentation of images from a ROS topic. It supports multiple prompt modes, including segmenting everything, and using text, points, or bounding boxes as prompts.

## Installation & Build

1.  **Clone the Package**:
    ```bash
    cd ~/your_catkin_ws/src
    git clone https://github.com/Michi-Tsubaki/fastsam_ros.git
    ```

2.  **Install System Dependencies**:
    ```bash
    sudo apt-get update
    cd ~/your_catkin_ws/src
    rosdep update --include-eol-distro
    rosdep install --from-path . -i -r -y
    ```

3.  **Build the Workspace**:
    Build the package using `catkin build`.
    This will automatically create a Python virtual environment, install the dependencies from `requirements.txt`, and download the `FastSAM-s.pt` model into the `models` directory.
    ```bash
    cd ~/your_catkin_ws/
    source devel/setup.bash #or source /opt/ros/<ROS_DISTRO>/setup.bash
    catkin build fastsam_ros
    ```

4.  **Source the Workspace**:
    Source your workspace's `setup.bash` file to make the node available in your environment.
    ```bash
    source ~/your_catkin_ws/devel/setup.bash
    ```

## 3. Usage

This package includes several launch files, one for each prompt mode. Make sure a camera driver node (or a rosbag) is publishing images to the topic specified in the `image_topic` argument.

### Common Launch Arguments

* `model_name` (string, default: `FastSAM-s.pt`): The name of the model file located in the `models` directory.
* `image_topic` (string, default: `/camera/image_raw`): The input image topic to subscribe to.
* `device` (string, default: `cpu`): The device to run inference on. Can be set to `cuda:0`, `cuda:1`, etc., for GPU acceleration.

### Launching in "Everything" Mode
This mode segments all objects detected in the image.

```bash
roslaunch fastsam_ros fastsam_everything.launch
```

### Launching in "Text" Mode
This mode segments objects described by a text prompt. The prompt is set in the launch file.

```bash
roslaunch fastsam_ros fastsam_text.launch
```
You can change the text prompt by editing the text_prompt param in fastsam_text.launch.

### Launching in "Point" Mode
This mode segments the object located at a given point.

1. Start the node
```bash
roslaunch fastsam_ros fastsam_point.launch
```

2. Publish a point prompt
```bash
rostopic pub /fastsam/point_prompt geometry_msgs/PointStamped "header:
  seq: 0
  stamp:
    secs: 0
    nsecs: 0
  frame_id: 'camera_frame'
point:
  x: 200.0
  y: 200.0
  z: 0.0" --rate 100
```

### Launching in "Bounding Box" Mode
1.
```bash
roslaunch fastsam_ros fastsam_bbox.launch
```

2. Publish a bounding box promp

```bash
rostopic pub /fastsam/bbox_prompt vision_msgs/BoundingBox2DArray "header:
  stamp: now
  frame_id: 'camera_frame'
boxes:
- center:
    x: 480
    y: 570
  size_x: 80
  size_y: 260" -1
```