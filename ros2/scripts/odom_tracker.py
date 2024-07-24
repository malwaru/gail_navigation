#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
import math
from collections import deque
from time import time

class OdomTracker(Node):
    def __init__(self):
        super().__init__('odom_tracker_node')
        self.subscriber_odom = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            10)
        self.pubisher_truncated_status = self.create_publisher(
            Bool,'/truncate_status', 10)
        self.positions = deque()
        self.last_time = time()
        self.declare_parameter('window_size', 10.0)  # seconds
        self.window_size = self.get_parameter('window_size').get_parameter_value().double_value
        self.msg_truncated= Bool()
        self.msg_truncated.data = False

    def odom_callback(self, msg):
        current_time = time()
        current_position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        
        # Append the current position and time to the deque
        self.positions.append((current_time, current_position))
        
        # Remove old positions that are outside the window size
        while self.positions and (current_time - self.positions[0][0] > self.window_size):
            self.positions.popleft()
        
        # Calculate the total distance travelled in the window
        total_distance = 0.0
        if len(self.positions) > 1:
            for i in range(1, len(self.positions)):
                total_distance += self.distance(self.positions[i-1][1], self.positions[i][1])
        
        # Calculate the average distance
        average_distance = total_distance / self.window_size
        
        # self.get_logger().info(f'Average distance travelled in the last {self.window_size} seconds: {average_distance:.2f} meters')
        if average_distance < 0.1:
            self.get_logger().info(f'Average distance in the last {self.window_size} seconds: {average_distance:.2f} meters \n Robot is stuck or crased \n truncating the episode')
            self.msg_truncated.data = True
            self.pubisher_truncated_status(self.msg_truncated)

        else:
            self.msg_truncated.data = False
            self.pubisher_truncated_status(self.msg_truncated)



    def distance(self, pos1, pos2):
        return math.sqrt((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2)

def main(args=None):
    rclpy.init(args=args)
    node = OdomTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
