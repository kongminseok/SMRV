# Efficient Human Detection Leveraging YOLOv8 Nano on ROS: A Comparative Study of RGB and RGB-D Cameras
[Daye Lee](https://github.com/Daye-Lee18), [Jiho Park](https://github.com/zihos), [Minseok Kong](https://github.com/kongminseok)


## Installation
### Prerequisites
- Python 3.x (tested on 3.8.10, Ubuntu 20.04)
- ROS (test ond ROS neotic)

- catkin workspace
    
    ```bash
    cd ~
    mkdir -p ~/catkin_ws/src
    cd ~/catkin_ws
    catkin_make
    source devel/setup.bash
    ```
    
- conda install
    
    ```bash
    wget https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-x86_64.sh
    chmod +x Anaconda3-2023.03-Linux-x86_64.sh
    ./Anaconda3-2023.03-Linux-x86_64.sh
    source ~/.bashrc
    ```
    
- camera package install
    
    https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md
    
    ```bash
    sudo mkdir -p /etc/apt/keyrings
    curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null
    
    echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo `lsb_release -cs` main" | \
    sudo tee /etc/apt/sources.list.d/librealsense.list
    sudo apt-get update
    
    sudo apt-get install librealsense2-dkms
    sudo apt-get install librealsense2-utils
    
    sudo apt-get install librealsense2-dev
    sudo apt-get install librealsense2-dbg
    
    # to verify the installation
    realsense-viewer
    ```
    
    **ddynamic_reconfigure**
    
    ```bash
    cd ~/catkin_ws/src
    git clone https://github.com/pal-robotics/ddynamic_reconfigure
    cd ..
    catkin_make
    ```
    
    **realsense-ros**
    
    ```bash
    cd ~/catkin_ws/src
    git clone https://github.com/IntelRealSense/realsense-ros.git
    cd realsense-ros/
    git checkout `git tag | sort -V | grep -P "^2.\d+\.\d+" | tail -1`
    cd ..
    catkin_init_workspace
    cd ..
    catkin_make clean
    catkin_make -DCATKIN_ENABLE_TESTING=False -DCMAKE_BUILD_TYPE=Release
    catkin_make install
    source devel/setup.bash 
    
    # to verify the installation
    roslaunch realsense2_camera rs_camera.launch
    
    ```
    
    for camera setting, edit `rs_camera.launch` . change the width and height of both color and depth image. set the fps `30`
    
    download the `HighAccuracyPreset.json` file and copy the path of the file and modify the launch file above.
    
    **camera node run with the preset:**
    
    `roslaunch realsense2_camera rs_camera.launch depth_width:=640 depth_height:=480 color_width:=640 color_height:=480 json_file_path:=/home/testzio/catkin_ws/src/realsense-ros/HighAccuracyPreset.json`
    
- git clone
    
    ```bash
    cd ~
    git clone https://github.com/zihos/iitp.git
    # enter your git username
    # get the developer token from the git setting and enter the token as a password
    
    ```
    
- For RGB package
    
    ```jsx
    mv ~/iitp/RGB ~/catkin_ws/src
    cd ~/catkin_ws/src/RGB/src
    chmod +x mobilenet_node.py 
    chmod +x video_publisher.py
    chmod +x yolov8_node.py
    ```
    
- conda env & dependency
    
    ```bash
    conda create -n ros python=3.8.10
    conda activate ros
    
    # pip install 
    cd ~/catkin_ws
    sudo apt install python3-pip
    # cpu-only version pytorch 
    python3 --version # python version check 
    pip install networkx==2.5.1
    pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cpu
    
    # install tensor flow
    pip install --upgrade pip
    pip install testresources
    pip install --upgrade setuptools
    pip install ultralytics
    pip install tensorflow==2.10.0
    
    cd ~/catkin_ws/src/RGB
    pip install -r requirements.txt
    pip install opencv-python==4.9.0.80
    sudo apt-get install ros-noetic-cv-bridge
    pip install --upgrade python-dateutil
    ```
    
- Run the RGB package
    - terminal 1 (camera node up)
    
    ```jsx
    cd ~/catkin_ws
    catkin_make
    source devel/setup.bash
    
    roslaunch realsense2_camera rs_camera.launch depth_width:=640 depth_height:=480 color_width:=640 color_height:=480 json_file_path:=/home/testzio/catkin_ws/src/realsense-ros/HighAccuracyPreset.json
    
    ```
    
    - terminal 2 (detection node up)
    
    ```jsx
    cd ~/catkin_ws
    catkin_make
    source devel/setup.bash
    roslaunch RGB my_launch.launch 
    ```
    
- For RGBD package
    
    ```jsx
    cd ~/iitp
    cp -r rgbd ~/catkin_ws/src/
    cd ~/catkin_ws/src/rgbd/src
    chmod +x rgbd_detection.py
    chmod +x subscriber.py
    
    pip install rospkg
    pip install empy==3.3.4
    pip install catkin_pkg
    ```
    
- Run the RGBD package
    - terminal 1 (camera node up)
    
    ```jsx
    cd ~/catkin_ws
    catkin_make
    source devel/setup.bash
    
    roslaunch realsense2_camera rs_camera.launch depth_width:=640 depth_height:=480 color_width:=640 color_height:=480 json_file_path:=/home/testzio/catkin_ws/src/realsense-ros/HighAccuracyPreset.json
    
    ```
    
    - terminal 2 (detection node up)
    
    ```jsx
    cd ~/catkin_ws
    catkin_make
    source devel/setup.bash
    roslaunch rgbd yolov8_ros.launch 
    ```


---
</hr>

## Acknowledgement
This work was supported by Institute of Information &amp; communications Technology Planning &amp; Evaluation (IITP) grant funded by the Korea government(MSIT) (RS-2022-00143911, AI Excellence Global Innovative Leader Education Program)
