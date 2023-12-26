import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image, Imu
from nav_msgs.msg import Odometry
import h5py
import numpy as np
from cv_bridge import CvBridge

class BagReaderNode(Node):
    def __init__(self):
        super().__init__('bag_reader_node')

        # Define the HDF5 file
        self.hdf5_file = h5py.File("output_data.h5", "w")

        # Define datasets for RGB, IMU, and odometry data
        self.rgb_dataset = self.hdf5_file.create_dataset("rgb_data", (0,), maxshape=(None,), dtype='u1', chunks=True)
        self.imu_dataset = self.hdf5_file.create_dataset("imu_data", (0, 6), maxshape=(None, 6), dtype='f')
        self.odom_dataset = self.hdf5_file.create_dataset("odom_data", (0, 7), maxshape=(None, 7), dtype='f')

        # Create a bridge between ROS Image messages and OpenCV images
        self.bridge = CvBridge()

        # Create a timer to read bag data every 0.5 seconds
        self.create_timer(0.5, self.read_bag_data)

    def read_bag_data(self):
        bag_file_path = 'your_ros2_bag_file.bag'  # Replace with the path to your ROS2 bag file
        bag = rosbag.Bag(bag_file_path)

        for topic, msg, t in bag.read_messages(topics=['camera', 'imu', 'odom/filtered']):
            if topic == 'camera':
                # Convert ROS Image to NumPy array
                rgb_data = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

                # Append the data to the HDF5 dataset
                self.rgb_dataset.resize(self.rgb_dataset.shape[0] + 1, axis=0)
                self.rgb_dataset[-1] = np.array(rgb_data.flatten(), dtype=np.uint8)

            elif topic == 'imu':
                # Extract IMU data and append to the HDF5 dataset
                imu_data = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
                            msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
                self.imu_dataset.resize(self.imu_dataset.shape[0] + 1, axis=0)
                self.imu_dataset[-1] = imu_data

            elif topic == 'odom/filtered':
                # Extract odometry data and append to the HDF5 dataset
                odom_data = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z,
                             msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z,
                             msg.twist.twist.angular.z]
                self.odom_dataset.resize(self.odom_dataset.shape[0] + 1, axis=0)
                self.odom_dataset[-1] = odom_data

        # Close the ROS bag
        bag.close()

    def destroy_node(self):
        # Close the HDF5 file when the node is destroyed
        self.hdf5_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    bag_reader_node = BagReaderNode()

    rclpy.spin(bag_reader_node)

    bag_reader_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
