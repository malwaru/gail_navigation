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

        self.depth_image_range=(0.05,50.0)

        self.depth_image=None
        self.rgb_image=None  
        self.cv_bridge = CvBridge() 
        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", self.show_depth)

    
    def transform_to_int8(self,arr,old_max=50.0):
        """
        Transform float 32 depth array to unsigned int 8 depth array.

        Args:
            arr (numpy.ndarray): The 2D array to be remapped.
            old_max (float): The maximum depth value 
                            Default max depth is 50.0 meters
            

        Returns:
            numpy.ndarray: The remapped 2D array with values as integers.
        """
        # Check if any value in the array is infinity and replace it with the maximum value
        arr = np.where(np.isinf(arr), old_max, arr)
        arr = arr/ old_max
         # normalize the data to 0 - 1
        rescaled_arr = 255 * arr # Now scale by 255
        return rescaled_arr.astype(np.int8)



    def depth_callback(self,msg):
        '''
        Receive postion of the tracked person 
        
        '''
        self.get_logger().info('Received depth image')
        depth_image=self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        ## The scale of depth pixels is 0.001 |  16bit depth, one unit is 1 mm 
        #  taken from data sheet        
        self.depth_image=self.transform_to_int8(depth_image)

    def image_callback(self,msg):
        '''
        Receive postion of the tracked person 
        
        '''
        self.rgb_image=self.cv_bridge.imgmsg_to_cv2(msg, 
                                                    desired_encoding='passthrough')
        
        cv2.imshow("Image", self.rgb_image)

            # Set mouse callback to show depth value
        cv2.setMouseCallback("Image", self.show_depth)
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):  # Press 'q' to exit
                break

        # cv2.waitKey(0)
        cv2.destroyAllWindows()

    def depth_probe(self):
        '''
        Get the depth of the tracked person
        
        '''
        if self.depth_image is not None:
            cv2.namedWindow("Image")
            cv2.imshow("Image", self.rgb_image)

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
            depth_value = self.depth_image[y, x]
            temp_img = copy.deepcopy(self.rgb_image)
            cv2.circle(temp_img, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(temp_img, f"Depth: {depth_value}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.imshow("Image",temp_img)



   



def main(args=None):
    rclpy.logging._root_logger.log(
        'Starting depth probe ...',
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
