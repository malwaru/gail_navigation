#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import  Image
from rclpy.logging import LoggingSeverity
import numpy as np
from cv_bridge import CvBridge
import cv2
import copy
class DepthProbe(Node):
    def __init__(self) -> None:
        super().__init__('depth_probe')
        self.subscriber_depth_image = self.create_subscription(
                                                Image,
                                                '/framos/depth/image_raw',
                                                self.depth_callback,
                                                10)
        self.subscriber_depth_image  # prevent unused variable warning

        self.subscriber_rgb_image = self.create_subscription(
                                                Image,
                                                '/framos/image_raw',
                                                self.image_callback,
                                                10)
        self.subscriber_rgb_image  # prevent unused variable warning
       
        # Publisher to pubsish person depth

        self.depth_image=None
        self.rgb_image=None  
        self.cv_bridge = CvBridge() 

    
 
    def depth_callback(self,msg):
        '''
        Receive postion of the tracked person 
        
        '''
        depth_image=self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
        ## The scale of depth pixels is 0.001 |  16bit depth, one unit is 1 mm 
        #  taken from data sheet
        self.depth_image=np.array(depth_image,dtype=np.uint16)*0.001


    def image_callback(self,msg):
        '''
        Receive postion of the tracked person 
        
        '''
        self.rgb_image=self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def depth_probe(self):
        '''
        Get the depth of the tracked person
        
        '''
        if self.depth_image is not None:
            cv2.namedWindow("Image")
            cv2.imshow("Image", self.old_img)

            # Set mouse callback to show depth value
            cv2.setMouseCallback("Image", self.show_depth)
            while True:
              key = cv2.waitKey(1) & 0xFF
              if key == ord("q"):  # Press 'q' to exit
                  break

            # cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            pass

    def show_depth(self,event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            depth_value = self.depth[y, x]
            temp_img = copy.deepcopy(self.image)
            cv2.putText(temp_img, "Depth: {:.2f}".format(depth_value), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.imshow("Image",temp_img)
            print(f"depth value {depth_value} at x {x} y {y}")


   



def main(args=None):
    rclpy.logging._root_logger.log(
        'Starting navigation command velocity publication ...',
        LoggingSeverity.INFO
    )
    rclpy.init(args=args)
    node = DepthProbe()
    rclpy.spin(node)
    # Destroy the node explicitly  
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
