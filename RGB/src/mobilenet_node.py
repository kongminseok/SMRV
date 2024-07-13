#!/usr/bin/env python3
from typing import List, Dict
import csv

import rospy
from std_msgs.msg import Header
from cv_bridge import CvBridge
import rospkg
import tensorflow as tf
import os
import numpy as np
import cv2

from tensorflow.keras.applications import *
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *

from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from RGB.msg import BoundingBox2D
from RGB.msg import Detection2D
from RGB.msg import Detection2DArray
from RGB.msg import Detection3D
from RGB.msg import Detection3DArray
from std_srvs.srv import SetBool, SetBoolResponse

import sys 
from openvino.runtime import Core
# import function from other files
# ROSpack + setup.py + catkin_install_python() and catkin_python_setup()
# https://answers.ros.org/question/411911/importing-python-filesfunctions-from-the-same-directory-as-simple-as-it-sounds/
rospack = rospkg.RosPack()
package_path = rospack.get_path('RGB')
sys.path.append(os.path.join(package_path, 'util'))

from detection3d_util import *


yolo_classes = ['Pedestrian', 'Cyclist', 'Car', 'motorcycle', 'airplane', 'Van', 'train', 'Truck', 'boat']
P2 = np.array([[718.856, 0.0, 607.1928, 45.38225], [0.0, 718.856, 185.2157, -0.1130887], [0.0, 0.0, 1.0, 0.003779761]])
dims_avg = {'Car': np.array([1.52131309, 1.64441358, 3.85728004]),
'Van': np.array([2.18560847, 1.91077601, 5.08042328]),
'Truck': np.array([3.07044968,  2.62877944, 11.17126338]),
'Pedestrian': np.array([1.75562272, 0.67027992, 0.87397566]),
'Person_sitting': np.array([1.28627907, 0.53976744, 0.96906977]),
'Cyclist': np.array([1.73456498, 0.58174006, 1.77485499]),
'Tram': np.array([3.56020305,  2.40172589, 18.60659898])}



class Detection3DNode:
    def __init__(self, real_time):
        rospy.init_node('detection_3d_node', anonymous=True)
        
        self.current_detections_msg = None
        self.current_img = None
        self.enable = True
        
        # Load the mobilenet model 
        self.ie = Core()
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('RGB')
        self.publisher = rospy.Publisher('detections_3d', Detection3DArray, queue_size=10)
        self.service = rospy.Service('enable', SetBool, self.enable_cb)


        self.model_path = os.path.join(package_path, 'model', 'quantized_model_openvino','quantized_model.xml') # for quantized mobilenet model
        # self.model_path = os.path.join(package_path, 'model', 'mobilenetv2_weights.h5') # for original mobilenet model

        self.device = rospy.get_param('~device', 'cpu')
        self.threshold = rospy.get_param('~threshold', 0.5)
        self.enable = rospy.get_param('~enable', True)
        self.input_image_topic = rospy.get_param('~input_image_topic', '/detections')

        # for original mobilenet model 
        # try:
        #     self.bbox3d_model = load_model(self.model_path)
        #     # print(self.bbox3d_model.summary())
        #     rospy.loginfo('Model loaded successfully')
        # except Exception as e:
        #     rospy.logerr('Failed to load model: {}'.format(e))
        #     exit(1) 
        
        try:
           self.bbox3d_model = self.ie.read_model(model=self.model_path)
           self.compiled_model = self.ie.compile_model(model=self.bbox3d_model, device_name='CPU')
           self.input_layer = self.compiled_model.input(0)
           self.output_layers = self.compiled_model.outputs
           rospy.loginfo('3d detection Model loaded successfully')
        except Exception as e:
           rospy.logerr('Failed to load model: {}'.format(e))
           exit(1)  

        rospy.loginfo('Detection3DNode created and configured')

        # Subscriber
        rospy.Subscriber('/detections', Detection2DArray, self.detections_callback)
        
        if real_time =="True":
            rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        else:
            rospy.Subscriber('/camera/rgb/image_raw', Image, self.image_callback)

        self.cv_bridge = CvBridge()


    def detections_callback(self, detections_msg):
        self.current_detections_msg = detections_msg 
    
    def image_callback(self, img):
        self.current_img=img

    def enable_cb(self, request):
        self.enable = request.data
        response_message = "Enabled" if self.enable else "Disabled"
        rospy.loginfo(response_message)
        return SetBoolResponse(success=True, message=response_message)
    
    def preprocess_image(self, img, xmin, ymin, xmax, ymax):
        # Crop and preprocess the image
        crop = img[int(ymin):int(ymax), int(xmin):int(xmax)]
        crop = cv2.resize(crop, (224, 224))
        crop = crop.astype(np.float32)
        crop = crop / 255.0  # Normalize to [0,1]
        crop = np.expand_dims(crop, axis=0)  # Add batch dimension
        return crop
    
    def predict(self, img, xmin, ymin, xmax, ymax):
        preprocessed_img = self.preprocess_image(img, xmin, ymin, xmax, ymax)
        result = self.compiled_model({self.input_layer.any_name: preprocessed_img})
        output = self.process_output(result)
        return output

    def process_output(self, result):
        # extract only needed data and reformatthem
        output1 = result[self.output_layers[0]]
        output2 = result[self.output_layers[1]]
        output3 = result[self.output_layers[2]]
        
        if output1.shape != (1, 6):
            raise ValueError(f"Unexpected shape for output1: {output1.shape}")
        if output2.shape != (1, 3):
            raise ValueError(f"Unexpected shape for output2: {output2.shape}")
        if output3.shape != (1, 6, 2):
            raise ValueError(f"Unexpected shape for output3: {output3.shape}")

        output1 = output1.reshape(1, -1)
        output2 = output2.reshape(1, -1)
        output3 = output3.reshape(1, -1, 2)

        final_output = [output2, output3, output1]
        return final_output
         

    def process_detections(self, img: Image, detections_msg: Detection2DArray):
        # self.bbox3d_model.summary()
        # Convert ROS Image message to OpenCV image
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(img, "bgr8")
        except CvBridgeError as e:
            print(e)
            return []

        DIMS = []
        bboxes = [] 
        results = []
        distances_csv = []
  
        for msg in detections_msg.detections:

            xmin = msg.bbox.center.position.x - (msg.bbox.size.x /2)
            ymin = msg.bbox.center.position.y - (msg.bbox.size.y /2)
            xmax = msg.bbox.center.position.x + (msg.bbox.size.x /2)
            ymax = msg.bbox.center.position.y + (msg.bbox.size.y /2)
            objID = msg.class_id
            
            # Model prediction
            prediction = self.predict(cv_image, xmin, ymin, xmax, ymax) #for openvino
            # prediction = self.bbox3d_model.predict(self.preprocess_image(cv_image, xmin, ymin, xmax, ymax), verbose = 0) # before quantization
            
            rospy.loginfo("Prediction: %s", prediction)
            
            dim = prediction[0][0]
            bin_anchor = prediction[1][0]
            bin_confidence = prediction[2][0]
            bin_size = 6

            ###refinement dimension
            try:
                # dim += dims_avg[str(yolo_classes[int(objID.cpu().numpy())])] + dim
                dim += dims_avg[str(yolo_classes[int(objID)])] + dim
                DIMS.append(dim)
            except:
                dim = DIMS[-1]
            
            bbox_ = [int(xmin), int(ymin), int(xmax), int(ymax)]
            
            frame = cv_image.copy()
            ###### 여기서 frame이 "전체" 이미지 맞겠지? 
            theta_ray = calc_theta_ray(frame, bbox_, P2)
            # update with predicted alpha, [-pi, pi]
            alpha = recover_angle(bin_anchor, bin_confidence, bin_size)
            alpha = alpha - theta_ray

            # initialize object container
            obj = KITTIObject()
            obj.name = str(yolo_classes[int(msg.class_id)]) 
            rospy.loginfo(str(yolo_classes[int(msg.class_id)]) )
            
            obj.truncation = float(0.00)
            obj.occlusion = int(-1)
            obj.xmin, obj.ymin, obj.xmax, obj.ymax = int(bbox_[0]), int(bbox_[1]), int(bbox_[2]), int(bbox_[3])
            bin_size=6
            obj.alpha = alpha
            obj.h, obj.w, obj.l = dim[0], dim[1], dim[2]
            
            # compute orientation 
            obj.rot_global, rot_local = compute_orientation(P2, obj)
            # 3D information on the object 
            
            # obj.tz = the distance between the camera and the object 
            obj.tx, obj.ty, obj.tz = translation_constraints(P2, obj, rot_local)

            # 3D position and orientation of the bounding box center: Pose center
            # total size of the bounding box, in meters, surrounding the object's center Vector3 size
            # frame reference string frame_id
            bbox3d = Detection3D()

            bbox3d.header = Header()
            bbox3d.header.frame_id = "base_link"
            
            bbox3d.class_name = msg.class_name
            bbox3d.confidence = bin_confidence
            bbox3d.bbox_3d.cx = obj.tx
            bbox3d.bbox_3d.cy = obj.ty
            bbox3d.bbox_3d.cz = obj.tz
            
            results.append(bbox3d)

            rospy.loginfo(f"estimated depth: {obj.tz} m")
            distances_csv.append(round(obj.tz, 2))


            cv2.rectangle(cv_image, (bbox_[0], bbox_[1]), (bbox_[2], bbox_[3]), (255, 0, 0), 2) 
            bbox_text = f"estimated depth: {obj.tz} m"
            cv2.putText(cv_image, bbox_text, (bbox_[0], bbox_[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        time_ = rospy.Time.now()


        return results


    def run(self):
        rate = rospy.Rate(2)  # Adjust the rate as necessary
        while not rospy.is_shutdown():
            if self.enable and self.current_detections_msg is not None and self.current_img is not None:
                results = self.process_detections(self.current_img, self.current_detections_msg)
                if results:
                    bbox3d_array = Detection3DArray()
                    bbox3d_array.header = Header()
                    bbox3d_array.header.stamp = rospy.Time.now()
                    bbox3d_array.header.frame_id = "base_link"
                    bbox3d_array.detections = results
                    self.publisher.publish(bbox3d_array)
                    # rospy.loginfo("Published 3D bounding boxes")
            rate.sleep()


def get_args():
    import sys
    import argparse
    parser = argparse.ArgumentParser(
        description=""
    )

    # Required arguments
    parser.add_argument("--real_time",
                        type=str,
                        default="True",
                        help="Whether it is a real-time experiment with a RGB-D camera")

    return parser.parse_args(rospy.myargv()[1:])    
      

def main():
    try:
        opt = get_args()
        node = Detection3DNode(opt.real_time)
        node.run()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
    
    