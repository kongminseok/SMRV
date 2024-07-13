import pyrealsense2 as rs
import numpy as np

# RealSense pipeline setting
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# start stream 
profile = pipeline.start(config)

# rgb and depth stream profile
depth_sensor = profile.get_device().first_depth_sensor()
depth_stream_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
color_stream_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))

# get extrinsic params
extrinsics = depth_stream_profile.get_extrinsics_to(color_stream_profile)

# print rotation and translation matrix
print("Rotation Matrix:\n", np.array(extrinsics.rotation).reshape(3, 3))
print("Translation Vector:\n", np.array(extrinsics.translation))

# end pipeline
pipeline.stop()
