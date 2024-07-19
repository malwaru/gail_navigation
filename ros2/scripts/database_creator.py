#!/usr/bin/env python3
import io
import rclpy
from rclpy.node import Node
from rclpy.logging import LoggingSeverity
from geometry_msgs.msg import PointStamped, Twist,PoseStamped,TransformStamped
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image, NavSatFix, Imu
from nav_msgs.msg import Odometry
import h5py
import numpy as np
import PIL, PIL.Image
from cv_bridge import CvBridge
import cv2
from kris_envs.wrappers.utilities import img_resize
import glob
import os
import pathlib

class BagReaderNode(Node):
    def __init__(self):
        ''' Defines the bag reader node
        '''

        super().__init__('bag_reader_node')

        # Basic setting to use
        # Define the HDF5 file
        file_path=os.getcwd()+'/'+os.path.basename(os.getcwd())+'.hdf5'
        self.hdf5_file = h5py.File(file_path, "w")
        self.image_shape=(240,320)
        self.image_compression_ratio=1.0
        self.image_shape_resized=(int(self.image_shape[0]*int(self.image_compression_ratio)),
                                  int(self.image_shape[1]*int(self.image_compression_ratio)))
        self.bag_read_freq=1.0
        # Define datasets    
        self.images= self.hdf5_file.create_group("images")
        self.kris_dynamics= self.hdf5_file.create_group("kris_dynamics")
        self.odom_data=self.kris_dynamics.create_group("odom_data")
        self.rgb_dataset = self.images.create_dataset("rgb_data",
                                                      (1,self.image_shape_resized[0],self.image_shape_resized[1],3),
                                                        maxshape=(None,self.image_shape_resized[0],self.image_shape_resized[1],3),
                                                        dtype=np.uint8)
        self.depth_dataset = self.images.create_dataset("depth_data", 
                                                      (1,self.image_shape_resized[0],self.image_shape_resized[1]),
                                                        maxshape=(None,self.image_shape_resized[0],self.image_shape_resized[1]),
                                                        dtype=np.uint8)
        
        self.imu_dataset = self.kris_dynamics.create_dataset("imu_data",(1, 6), maxshape=(None, 6), dtype='f')
        #initialize the odom variables
        self.odoms_filtered=np.empty((1,7))
        self.odoms_wheel=np.empty((1,7))
    




        # Create a bridge between ROS Image messages and OpenCV images
        self.bridge = CvBridge()

        # Create a timer to read bag data every n seconds
        self.create_timer(self.bag_read_freq, self.read_bag_data)
        # self.future = self.create_future()  # Create a future to wait for the goal pose

        # The topic list to read from the bag file
        self.topics_list = [
                        "/clicked_point",
                        "/clock",
                        "/cmd_vel",
                        "/framos/camera_info",
                        "/framos/depth/camera_info",
                        "/framos/depth/image_raw",
                        "/framos/image_raw",
                        "/goal_pose",
                        "/gps/fix",
                        "/imu",
                        "/initialpose",
                        "/odometry/filtered",
                        "/odometry/wheel",
                        "/set_pose",
                        "/tf",
                        "/tf_static"
                    ]

        #Intiating the subscribers for the topics
        self.clicked_point_data = None
        self.clock_data = None
        self.cmd_vel_data = None
        self.camera_info_data = None
        self.depth_camera_info_data = None
        self.depth_image_raw_data = None
        self.image_raw_data = None
        self.goal_pose_data = None
        self.gps_fix_data = None
        self.imu_data = None
        self.initial_pose_data = None
        self.odometry_filtered_data = None
        self.odometry_wheel_data = None
        self.set_pose_data = None
        self.tf_data = None
        self.tf_static_data = None

        # Create subscribers for each topic
        self.clicked_point_sub = self.create_subscription(
            PointStamped, '/clicked_point', self.clicked_point_callback, 10)
        self.clock_sub = self.create_subscription(
            Clock, '/clock', self.clock_callback, 10)
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/framos/camera_info', self.camera_info_callback, 10)
        self.depth_camera_info_sub = self.create_subscription(
            CameraInfo, '/framos/depth/camera_info', self.depth_camera_info_callback, 10)
        self.depth_image_raw_sub = self.create_subscription(
            Image, '/framos/depth/image_raw', self.depth_image_raw_callback, 10)
        self.image_raw_sub = self.create_subscription(
            Image, '/framos/image_raw', self.image_raw_callback, 10)
        self.goal_pose_sub = self.create_subscription(
            PoseStamped, '/goal_pose', self.goal_pose_callback, 10)
        self.gps_fix_sub = self.create_subscription(
            NavSatFix, '/gps/fix', self.gps_fix_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu', self.imu_callback, 10)
        self.initial_pose_sub = self.create_subscription(
            PoseStamped, '/initialpose', self.initial_pose_callback, 10)
        self.odometry_filtered_sub = self.create_subscription(
            Odometry, '/odometry/filtered', self.odometry_filtered_callback, 10)
        self.odometry_wheel_sub = self.create_subscription(
            Odometry, '/odometry/wheel', self.odometry_wheel_callback, 10)
        self.set_pose_sub = self.create_subscription(
            PoseStamped, '/set_pose', self.set_pose_callback, 10)
        self.tf_sub = self.create_subscription(
            TransformStamped, '/tf', self.tf_callback, 10)
        self.tf_static_sub = self.create_subscription(
            TransformStamped, '/tf_static', self.tf_static_callback, 10)


    # callback functions for each topic
    def clicked_point_callback(self, msg):
        self.clicked_point_data = msg

    def clock_callback(self, msg):
        self.clock_data = msg

    def cmd_vel_callback(self, msg):
        self.cmd_vel_data = msg

    def camera_info_callback(self, msg):
        self.camera_info_data = msg

    def depth_camera_info_callback(self, msg):
        self.depth_camera_info_data = msg

    def depth_image_raw_callback(self, msg):
        self.depth_image_raw_data = self.bridge.imgmsg_to_cv2(msg, 
                                                desired_encoding='passthrough')

    def image_raw_callback(self, msg):
        self.image_raw_data = self.bridge.imgmsg_to_cv2(msg,
                                                desired_encoding='passthrough')
    def goal_pose_callback(self, msg):
        '''
        This function is called when the goal pose is available
        the goal pose is available only at the end of the episode
        therefore, we will create the odom and goal dataset here
        '''
        ## Since goal pose is collected only at the eng of the episode, 
        # we will create the all datasets here
        self.goal_pose_data = msg

        # Filter odom dataset

        self.odoms_filtered=np.vstack((self.odoms_filtered,self.odoms_filtered[-1,:]))
        self.odoms_filtered=np.delete(self.odoms_filtered,0,axis=0)
        self.get_logger().info("The odom shape is: {}".format(self.odoms_filtered.shape))
        self.odom_data.create_dataset("odom_data_filtered", data=self.odoms_filtered)
        print(f"\n \nThe odom shape  stage 1 is: {self.odoms_wheel.shape} \n")
        self.odoms_wheel=np.vstack((self.odoms_wheel,self.odoms_wheel[-1,:]))
        print(f"The odom shape  stage 2 is: {self.odoms_wheel.shape} \n")
        self.odoms_wheel=np.delete(self.odoms_wheel,0,axis=0)
        print(f"The odom shape  stage 3 is: {self.odoms_wheel.shape}\n \n")
        self.odom_data.create_dataset("odom_data_wheel", data=self.odoms_wheel)
        self.get_logger().info("The odom shape is: {}".format(self.odoms_wheel.shape))

        # Create goal dataset
        goal_data = [self.goal_pose_data.pose.position.x, 
                     self.goal_pose_data.pose.position.y, 
                     self.goal_pose_data.pose.position.z,
                     self.goal_pose_data.pose.orientation.x, 
                     self.goal_pose_data.pose.orientation.y,
                     self.goal_pose_data.pose.orientation.z,
                     self.goal_pose_data.pose.orientation.w]
        #Create the relative goal pose from each location
        # goal_data=np.repeat(goal_data,len(self.odoms_filtered),axis=0)
        # self.get_logger().info("The goal shape is: {}".format(goal_data.shape))
        # target=goal_data-self.odoms_filtered
        goal_data=np.repeat(goal_data,len(self.odoms_wheel),axis=0).reshape(len(self.odoms_wheel),7)
        self.get_logger().info("The goal shape is: {}".format(goal_data.shape))
        target=goal_data-self.odoms_wheel
        self.get_logger().info("The target shape is: {}".format(target.shape))
        self.odom_data.create_dataset("target_vector", data=target)
        self.destroy_node()
     

    def gps_fix_callback(self, msg):
        self.gps_fix_data = msg

    def imu_callback(self, msg):
        self.imu_data = msg

    def initial_pose_callback(self, msg):
        self.initial_pose_data = msg

    def odometry_filtered_callback(self, msg):
        self.odometry_filtered_data = msg

    def odometry_wheel_callback(self, msg):
        self.odometry_wheel_data = msg

    def set_pose_callback(self, msg):
        self.set_pose_data = msg

    def tf_callback(self, msg):
        self.tf_data = msg

    def tf_static_callback(self, msg):
        self.tf_static_data = msg
    
    def im2bytes(self,arrs,format='jpg'):
        '''
        Create a JPEG encoded image from a NumPy array
        '''
        if len(arrs.shape) == 4:
            return np.array([self.im2bytes(arr_i, format=format) for arr_i in arrs])
        elif len(arrs.shape) == 3:
            im = PIL.Image.fromarray(arrs.astype(np.uint8))
            with io.BytesIO() as output:
                im.save(output, format="jpeg")
                return output.getvalue()
        else:
            raise ValueError

    def bytes2im(self,arrs):
        '''
        Decode the JPEG encoded image to a NumPy array
        '''
        if len(arrs.shape) == 1:
            return np.array([self.bytes2im(arr_i) for arr_i in arrs])
        elif len(arrs.shape) == 0:
            return np.array(PIL.Image.open(io.BytesIO(arrs)))
        else:
            raise ValueError


    def read_bag_data(self):
        '''
        Read data from the bag file and append it to the HDF5 dataset
                
        '''    

        for topic in self.topics_list:
            if topic == '/framos/image_raw':
                # Convert image to numpy channel format               
                rgb_data=cv2.cvtColor(self.image_raw_data, cv2.COLOR_BGR2RGB)
                img_resized=img_resize(rgb_data,self.image_compression_ratio)
                # rgb_encoded = self.im2bytes(img_resized)

                # Append the data to the HDF5 dataset
                self.rgb_dataset.resize(self.rgb_dataset.shape[0] + 1, axis=0)
                self.rgb_dataset[-1] = img_resized
                
            if topic=="/framos/depth/image_raw":
                # The depth values are in metres
                # We do not convert the depth values to uint8                                         
                depth_data = self.depth_image_raw_data
                ## The scale of depth pixels is 0.001|  16bit depth, one unit is 1 mm | taken from data sheet 
                # depth_data = np.array(depth_data,dtype=np.uint16)*0.001           
                img_depth_resized=img_resize(depth_data,self.image_compression_ratio)

            # Append the data to the HDF5 dataset
                self.depth_dataset.resize(self.depth_dataset.shape[0] + 1, axis=0)
                self.depth_dataset[-1] = img_depth_resized
                

            elif topic == '/imu':
                # Extract IMU data and append to the HDF5 dataset
                imu_data = [self.imu_data.linear_acceleration.x, 
                            self.imu_data.linear_acceleration.y,
                            self.imu_data.linear_acceleration.z,
                            self.imu_data.angular_velocity.x, 
                            self.imu_data.angular_velocity.y,
                            self.imu_data.angular_velocity.z]
                self.imu_dataset.resize(self.imu_dataset.shape[0] + 1, axis=0)
                self.imu_dataset[-1,:] = imu_data

            elif topic == '/odoms/filtered':
                # Extract odometry data and append to the HDF5 dataset
                odom_data = [self.odometry_filtered_data.pose.pose.position.x, 
                             self.odometry_filtered_data.pose.pose.position.y, 
                             self.odometry_filtered_data.pose.pose.position.z,
                             self.odometry_filtered_data.pose.pose.position.z,
                             self.odometry_filtered_data.pose.pose.orientation.x, 
                             self.odometry_filtered_data.pose.pose.orientation.y,
                               self.odometry_filtered_data.pose.pose.orientation.z,
                             self.odometry_filtered_data.pose.pose.orientation.w]
                self.odoms_filtered=np.vstack((self.odoms_filtered,odom_data))
            
            elif topic=='/odometry/wheel':
                odom_data = [self.odometry_wheel_data.pose.pose.position.x, 
                             self.odometry_wheel_data.pose.pose.position.y, 
                             self.odometry_wheel_data.pose.pose.position.z,
                             self.odometry_wheel_data.pose.pose.orientation.x, 
                             self.odometry_wheel_data.pose.pose.orientation.y,
                             self.odometry_wheel_data.pose.pose.orientation.z,
                             self.odometry_wheel_data.pose.pose.orientation.w]
                self.odoms_wheel=np.vstack((self.odoms_wheel,odom_data))
             



    def destroy_node(self):
        # Close the HDF5 file when the node is destroyed
        self.hdf5_file.close()
        super().destroy_node()

        


def main(args=None):
    rclpy.logging._root_logger.log(
        'Starting recorindg to hdf5 format ...',
        LoggingSeverity.INFO
    )

    rclpy.init(args=args)

    bag_reader_node = BagReaderNode()

    rclpy.spin(bag_reader_node)

    bag_reader_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
