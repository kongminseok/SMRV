ubuntu20.04 (linux aarch64), ROS1 noetic, Intel CPU, python=3.8

0. Environment setup
```bash
# environment setup
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc

sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
sudo apt install python3-rosdep
sudo rosdep init
rosdep update
```

1. Setup and Install
```bash
# Creating a workspace for catkin
cd ~
source /opt/ros/noetic/setup.bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/
catkin_make
source devel/setup.bash
echo $ROS_PACKAGE_PATH 
# /home/youruser/catkin_ws/src:/opt/ros/kinetic/share  or /home/ubuntu/catkin_ws/src:/opt/ros/noetic/share

# github connect 
ls -al ~/.ssh
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
eval "$(ssh-agent -s)" 
ssh-add ~/.ssh/id_rsa
cat ~/.ssh/id_rsa.pub
# Go to GitHub, navigate to Settings > SSH and GPG keys > New SSH key, and paste the key.
git config --global user.name "Your Name"
git config --global user.email "your_email@example.com"

cd ~
git clone git@github.com:zihos/iitp.git
mv ~/iitp/RGB src
cd ~/catkin_ws/src/RGB/src
chmod +x mobilenet_node.py 
chmod +x video_publisher.py
chmod +x yolov8_node.py
```

2. Conda install in aarch64 linux
```
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
bash Miniforge3-Linux-aarch64.sh -b
source ~/miniforge3/etc/profile.d/conda.sh
conda init
source ~/.bashrc
conda config --set auto_activate_base false
source ~/.bashrc
# if you want to reactivate conda env, enter conda activate base
```

3. Dependency
```
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
python3 -m pip install tensorflow==2.10.0

cd ~/catkin_ws/src/RGB
pip install -r requirements.txt
sudo apt-get install --reinstall ros-noetic-cv-bridge
pip install --upgrade python-dateutil

nano ~/.bashrc
#Add the following lines at the end of the file:
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libstdc++.so.6
# exit 
source ~/.bashrc

cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

4. Run the code 

```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash

# Run all the node in rqt_graph:
roslaunch RGB my_launch.launch real_time:=True
# If real_time == True, in another terimnal
cd ./catkin_ws
rosbag play ${Data_directory} -d 1 -r 0.1 

# If real_time == False (== use the video as the input), open the `~/caktkin_ws/iitp/RGB/src/video_publisher.py` and change the video directory at a line number 14
roslaunch RGB my_launch.launch real_time:=False

# For debugging, run each node 
rosrun RGB ${파일name} real_time:=True
rosrun RGB video_publisher.py 
rosrun RGB yolov8_node.py real_time:=True
rosrun RGB mobilenet_node.py real_time:=True
```

