#!/usr/bin/env python3

import rospy
from rgbd.msg import DetectionArray  # 여기서 your_pkg는 DetectionArray 메시지가 정의된 패키지 이름으로 바꿔야 합니다.

class DetectionSubscriber:
    def __init__(self):
        rospy.init_node('detection_subscriber', anonymous=True)
        self.subscriber = rospy.Subscriber('/yolov8/detections', DetectionArray, self.callback)
        rospy.loginfo("DetectionSubscriber initialized and subscribed to /yolov8/detections")

    def callback(self, msg):
        rospy.loginfo(f"Received {len(msg.detections)} detections")
        for detection in msg.detections:
            rospy.loginfo(f"Detection: {detection}")

if __name__ == '__main__':
    try:
        DetectionSubscriber()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
