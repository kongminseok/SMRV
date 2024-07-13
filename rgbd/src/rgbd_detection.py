#!/usr/bin/env python3
import csv # remove it later
from typing import List, Dict
import os
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from message_filters import Subscriber, ApproximateTimeSynchronizer


from sensor_msgs.msg import Image
from std_msgs.msg import Header
from rgbd.msg import Detection, DetectionArray, BoundingBox2D
import rospkg

from std_srvs.srv import SetBool, SetBoolResponse
import time

from ultralytics import YOLO
from ultralytics.engine.results import Results

import math


class RGBDDetectionNode:
    def __init__(self):
        rospy.init_node('rgbd_detection', anonymous=True)
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('rgbd')

        # set the rate
        self.rate = rospy.Rate(3)  # 3 Hz (3 FPS)
        self.cv_bridge = CvBridge()


        # Subscribers for color and depth images
        self.color_sub = Subscriber("/camera/color/image_raw", Image)
        self.depth_sub = Subscriber("/camera/depth/image_rect_raw", Image)

        # ApproximateTimeSynchronizer to synchronize the topics
        self.ts = ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], queue_size=10, slop=0.5)
        self.ts.registerCallback(self.callback)

        self.color_image = None
        self.depth_image = None

        self.red = (0, 0, 255)  # red
        self.green = (0, 255, 0) # green
        self.thickness = 1

        # camera intrinsic
        # Focal lengths
        self.fx = int(612.9598388671875)
        self.fy = int(613.0750732421875)
        # Principal points
        self.cx = int(317.4980773925781)
        self.cy = int(244.09921264648438)

        # from the actual params from cameara_extrinsic.py
        self.rotation_matrix = np.array([[0.99990785,0.00959786,-0.00960118],[-0.00957087,0.99995011,0.00285352],[0.00962809,-0.00276137,0.99994981]])  
        self.translation_vector = np.array([0.01486797,0.00033487,0.0003925])

        self.maximum_detection_threshold_ = 50  # for center depth
        rospy.loginfo("RGBDDetectionNode Initialized")

        

        # Initialize the Yolo model (openvino)
        self.model_path = os.path.join(package_path, 'model', 'yolov8n_openvino_model')
        self.model = rospy.get_param('~model', 'yolov8n.pt')
        self.device = rospy.get_param('~device', 'cpu')
        self.threshold = rospy.get_param('~threshold', 0.5)
        self.enable = rospy.get_param('~enable', True)
        # self.fps = rospy.get_param('~fps', 5)
        self.yolo = YOLO(self.model_path)

         # Initialize the Yolo model (pytorch)
        # self.model = rospy.get_param('~model', 'yolov8n.pt')
        # self.yolo = YOLO(self.model)
        # self.yolo.fuse()

        rospy.loginfo('Yolov8model created and configured')

        self.publisher = rospy.Publisher('/yolov8/detections', DetectionArray, queue_size=10)


    def enable_cb(self, req):
        self.enable = req.data
        return SetBoolResponse(success=True, message="Enable State Updated")

    def callback(self, color_msg, depth_msg):
        try:
            self.color_image = self.cv_bridge.imgmsg_to_cv2(color_msg, "bgr8")
            self.depth_image = self.cv_bridge.imgmsg_to_cv2(depth_msg, "32FC1")
            self.process_image()
        except CvBridgeError as e:
            rospy.logerr(f"Callback error: {e}")

    def deproject_pixel_to_point(self, x, y, depth):
        """
        Calculate 3D point using 2D pixel point and depth value
        """
        X = (x - self.cx) * depth / self.fx
        Y = (y - self.cy) * depth / self.fy
        Z = depth
        return (X, Y, Z)
    
    # reference : https://github.com/IntelligentRoboticsLabs/gb_visual_detection_3d
    def new3d(self, x1, x2, y1, y2, depth_image: Image) -> None:
        center_x = (x1+x2)//2
        center_y = (y1+y2)//2
        depth_roi = depth_image[center_y-10:center_y+10, center_x-10:center_x+10]
        
        center_depth = np.mean(depth_roi[depth_roi>0])
        rospy.loginfo(f"center depth: {center_depth}")
        # rospy.loginfo(f"{depth_image.shape}, center: {center_x}, {center_y}, center depth: {center_depth}, depth sum: {np.sum(depth_image)}")
        
        if np.isnan(center_depth) or center_depth == 0:
            rospy.loginfo(f"center depth is zero")

        center_point = self.deproject_pixel_to_point(center_x, center_y, center_depth)

        maxx = maxy = maxz = -np.inf
        minx = miny = minz = np.inf

        for i in range(x1, x2):
            for j in range(y1, y2):
                point_depth = depth_image[j, i]

                if np.isnan(point_depth) or point_depth == 0:
                    continue

                point = self.deproject_pixel_to_point(i, j, point_depth)

                if abs(point[2] - center_point[2]) < self.maximum_detection_threshold_:
                    maxz = max(point[2], maxz)
                    minz = min(point[2], minz)
                
                maxx = max(point[0], maxx)
                maxy = max(point[1], maxy)
                
                minx = min(point[0], minx)
                miny = min(point[1], miny)
                    
        rospy.loginfo(f"new distance: {minz}, {maxz}, {((minz+maxz)/2)/1000.0:.2f}m")
        
        return ((minz+maxz)/2)/1000.0, center_depth

            

    
    def parse_boxes(self, results: Results, depth_image: Image) -> List[Dict]:
        boxes_list = []
        new3d_list = []

        for box_data in results.boxes:
            bbox_2d = BoundingBox2D()
            
            x1, y1, x2, y2 = map(int, box_data.xyxy[0])

            new_dist, center_depth = self.new3d(x1, x2, y1, y2, depth_image)
            new3d_list.append(new_dist)
            rospy.loginfo(f"new distance: {new_dist:.2f}")
            
            depth_crop = depth_image[y1:y2, x1:x2]
            if depth_crop.size == 0:
                continue
            z1 = np.min(depth_crop[depth_crop > 0]) / 1000.0  # convert depth values into meters
            z2 = np.median(depth_crop[depth_crop > 0]) / 1000.0  # convert depth values into meters
        
            # center of 2D bounding box
            bbox_2d.cx = (x1 + x2) / 2
            bbox_2d.cy = (y1 + y2) / 2
            # calculated z coordinate (distance)
            bbox_2d.cz = center_depth #(z1 + z2) / 2
            rospy.loginfo(f"min distance: {z1:.2f}m, max distance: {z2:.2f}m, min+max/2: {(z1 + z2) / 2:.2f}m")

            boxes_list.append({
                "bbox_2d": bbox_2d,
            })

        return boxes_list, new3d_list

    def process_image(self) -> None:

        if self.color_image is not None and self.depth_image is not None:
            try:
                # Normalize the depth image for display purposes
                depth_display = cv2.normalize(self.depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

                results = self.yolo.predict(
                    source=self.color_image,
                    verbose=False,
                    stream=False,
                    classes=[0],
                    conf=self.threshold,
                    device=self.device
                )
                
                results: Results = results[0].cpu()
                
                detection_msg = DetectionArray()
                detection_msg.header = Header()
                detection_msg.header.stamp = rospy.Time.now()

                height, width = self.depth_image.shape

                boxes = []
                distances_csv = []
                if results.boxes:
                    boxes, new_3d = self.parse_boxes(results, self.depth_image)
                    for i, box_data in enumerate(results.boxes):
                        detection = Detection()
                        detection.class_name = self.yolo.names[int(box_data.cls)]
                        detection.confidence = float(box_data.conf)
                        detection.bbox_2d = boxes[i]["bbox_2d"]
                        human_dist = new_3d[i]

                        xyxy = box_data.xyxy[0]
                        xmin, ymin, xmax, ymax = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                        xmin = max(0, xmin)
                        ymin = max(0, ymin)
                        xmax = min(width, xmax)
                        ymax = min(height, ymax)
                        start_point = (xmin, ymin)
                        end_point = (xmax, ymax)

                        # for image visualization
                        cv2.rectangle(self.color_image, start_point, end_point, self.red, 2) 
                        bbox_text = f"new_3d: {human_dist:.2f}m, cz: {detection.bbox_2d.cz:.2f}m"
                        distances_csv.append([round(human_dist,2), round(detection.bbox_2d.cz,2)])  # for results recording
                        cv2.putText(self.color_image, bbox_text, (start_point[0], start_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.green, 2)

                        detection_msg.detections.append(detection)
                
                self.publisher.publish(detection_msg)
                time_ = rospy.Time.now()

                # Display the combined images
                # combined_image = np.vstack((self.color_image, depth_display))
                # cv2.imshow("RGB and Depth Image", combined_image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    rospy.signal_shutdown("User requested shutdown")
                
                del results
                del boxes
                
            except Exception as e:
                rospy.logerr(f"Error in show_images: {e}")


            

if __name__ == "__main__":
    try:
        node = RGBDDetectionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()