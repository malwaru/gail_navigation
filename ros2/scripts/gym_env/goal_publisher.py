#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge 
from rclpy.logging import LoggingSeverity
import cv2
# from sort import Sort 
import numpy as np


class GoalPublisher(Node):

    def __init__(self, device=0):
        super().__init__('web_stream')
        self.publisher_webstream = self.create_publisher(Image,
                                                         '/camera/color/image_raw',
                                                         10)

        # Other variables
        self._cvbridge = CvBridge()
        self._robot_stream = None
        self.camera = cv2.VideoCapture(device)
        self.stream_camera()
        

    def stream_camera(self, device=0):
        '''
        Stream
        '''
        
        _,stream = self.camera.read()
        # cv2.imshow("Tracking",stream)
        ros_image=self._cvbridge.cv2_to_imgmsg(stream)    
        self.publisher_webstream.publish(ros_image)
        cv2.waitKey(1)



def main(args=None):
    rclpy.logging._root_logger.log(
        'Starting test script  ...',
        LoggingSeverity.INFO
    )
    rclpy.init(args=args)

    goal_publisher=GoalPublisher()

    rclpy.spin(goal_publisher)

    # Destroy the node explicitly  
    goal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()