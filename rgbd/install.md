
```bash
# Create a ROS Workspace
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/
catkin_make
source devel/setup.bash

# Setup
cd ~/catkin_ws/
git clone https://github.com/zihos/iitp.git
cd iitp
pip install -r requirements.txt
mv rgbd src
cd ~/catkin_ws/src/rgbd/src
chmod +x rgbd_detection.py 
chmod +x subscriber.py # dummy node
cd ~/catkin_ws
catkin_make
source devel/setup.bash

# Run
roslaunch rgbd yolov8_ros.launch

# Debug rgbd_detection.py node
roscore
cd ~/catkin_ws
rosrun rgbd rgbd_detection.py
```
