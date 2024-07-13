import pyrealsense2 as rs

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
profile = pipeline.start(config)

# Get the sensor
depth_sensor = profile.get_device().first_depth_sensor()
color_sensor = profile.get_device().query_sensors()[1]

# Get depth stream intrinsics
depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
print("Depth intrinsics: ")
print(f"  Width: {depth_intrinsics.width}")
print(f"  Height: {depth_intrinsics.height}")
print(f"  PPX: {depth_intrinsics.ppx}")
print(f"  PPY: {depth_intrinsics.ppy}")
print(f"  Focal Length X: {depth_intrinsics.fx}")
print(f"  Focal Length Y: {depth_intrinsics.fy}")
print(f"  Distortion Coefficients: {depth_intrinsics.coeffs}")

# Get color stream intrinsics
color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
print("Color intrinsics: ")
print(f"  Width: {color_intrinsics.width}")
print(f"  Height: {color_intrinsics.height}")
print(f"  PPX: {color_intrinsics.ppx}")
print(f"  PPY: {color_intrinsics.ppy}")
print(f"  Focal Length X: {color_intrinsics.fx}")
print(f"  Focal Length Y: {color_intrinsics.fy}")
print(f"  Distortion Coefficients: {color_intrinsics.coeffs}")

# Stop streaming
pipeline.stop()
