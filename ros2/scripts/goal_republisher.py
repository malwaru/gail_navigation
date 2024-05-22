#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.logging import LoggingSeverity
from geometry_msgs.msg import PoseStamped

class GoalStream(Node):
    def __init__(self):
        '''
        Subcribe to the sporadic output named projected_map from octomap_server
        and publish a map topic called use_map to be used by the navigation stack 
        '''
        super().__init__('goal_stream')
        self.publisher_goalstream = self.create_publisher(PoseStamped,
                                                         '/target_goal',
                                                         10)
        
        self._subscriber_static_goal = self.create_subscription(
                                                PoseStamped,
                                                '/goal_pose',
                                                self.static_goal_callback,
                                                10)
        self._subscriber_static_goal

        # Other variables
        self.get_logger().info(f'continually publishing goal stream ...')
        self._goal_stream = None
        self.first_received=False
       
        

    def static_goal_callback(self,msg):
        '''
        republish static map as a continous stream
        '''
        self._goal_stream=msg
        self.first_received=True
        if self.first_received:
            self.publish_continous_goal()
   

        

    def publish_continous_goal(self):
        '''
        Stream of goals
        '''        
        # Has to change the while true function to more practical 
        while True:
            self.publisher_goalstream.publish(self._goal_stream)


def main(args=None):
    rclpy.logging._root_logger.log(
        'Starting contiual map script  ...',
        LoggingSeverity.INFO
    )
    rclpy.init(args=args)
    map_node = GoalStream() 
    rclpy.spin(map_node)
    # Destroy the node explicitly  
    map_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
