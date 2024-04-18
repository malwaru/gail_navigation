#!/usr/bin/env python
import os
import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetPhysicsProperties, SpawnEntity, DeleteEntity
from geometry_msgs.msg import Pose,PoseStamped
from ament_index_python.packages import get_package_share_directory
import numpy as np
import random
from xml.etree import ElementTree
from rclpy.logging import LoggingSeverity



class GazeboConnection(Node):
    '''
    The class that connect the gym environment to the gazebo simulation    
    
    '''

    def __init__(self):
        super().__init__('gazebo_connection_node')

        self.unpause = self.create_client(Empty, '/unpause_physics')
        self.pause = self.create_client(Empty, '/pause_physics')
        self.reset_proxy = self.create_client(Empty, '/reset_simulation')
        self.spawn_model = self.create_client(SpawnEntity, '/spawn_entity')
        self.delete_model = self.create_client(DeleteEntity, '/delete_entity')
        self.goal_pose_pub = self.create_publisher(PoseStamped,'/goal_pose',10)
        self.publisher_subgoal_pose = self.create_publisher(PoseStamped,
                                                '/subgoal_pose',                                                
                                                10)

        # Setup the Gravity Control system
        # service_name = '/gazebo/set_parameters'
        # self.set_physics = self.create_client(SetPhysicsProperties, service_name)
        # self.get_logger().info("Waiting for service " + str(service_name))
        # while not self.set_physics.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().info("Service not available, waiting again...")
        # self.get_logger().info("Service Found " + str(service_name))

        ## Parameters for robot spawn

        self.timeout = 30.0
        self.entity='kris'
        self.package_name = 'kris_description'
        self.pkg_share = get_package_share_directory(self.package_name)
        urdf_dir=os.path.join(self.pkg_share, 'urdf')
        self.urdf = os.path.join(urdf_dir, 'KRIS.urdf')

        # self.init_values()
        # We always pause the simulation, important for legged robots learning
        # self.pause_sim()
        # self.spawn_robot()

    def pause_sim(self):
        request = Empty.Request()
        future = self.pause.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info("/gazebo/pause_physics service call successful")
        else:
            self.get_logger().error("/gazebo/pause_physics service call failed")

    def unpause_sim(self):
        '''
        Unpause the simulation
        '''
        request = Empty.Request()
        future = self.unpause.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info("/gazebo/unpause_physics service call successful")
        else:
            self.get_logger().error("/gazebo/unpause_physics service call failed")

    def reset_sim(self):
        '''
        Reset the simualtion
        
        '''
        request = Empty.Request()
        future = self.reset_proxy.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info("/gazebo/reset_simulation service call successful")
        else:
            self.get_logger().error("/gazebo/reset_simulation service call failed")

    def reset_world(self):
        '''
        Reset the simulation and spawn the robot again        
        '''
        self.reset_sim()  
        self.delete_entity()
        self.spawn_robot()

    def init_values(self):
        # self.reset_sim()

        # self._time_step = Float64(data=0.001)
        # self._max_update_rate = Float64(data=1000.0)

        # self._gravity = Vector3()
        # self._gravity.x = 0.0
        # self._gravity.y = 0.0
        # self._gravity.z = 0.0

        # self._ode_config = ODEPhysics()
        # self._ode_config.auto_disable_bodies = False
        # self._ode_config.sor_pgs_precon_iters = 0
        # self._ode_config.sor_pgs_iters = 50
        # self._ode_config.sor_pgs_w = 1.3
        # self._ode_config.sor_pgs_rms_error_tol = 0.0
        # self._ode_config.contact_surface_layer = 0.001
        # self._ode_config.contact_max_correcting_vel = 0.0
        # self._ode_config.cfm = 0.0
        # self._ode_config.erp = 0.2
        # self._ode_config.max_contacts = 20

        # self.update_gravity_call()
        return NotImplementedError

    def _spawn_entity(self, entity_xml, initial_pose, timeout):
        if timeout < 0:
            self.get_logger().error('spawn_entity timeout must be greater than zero')
            return False
        self.get_logger().info(
            'Waiting for service /spawn_entity, timeout = %.f' % timeout)
        self.get_logger().info('Waiting for service /spawn_entity')
        # client = self.create_client(SpawnEntity, '/spawn_entity')
        if self.spawn_model.wait_for_service(timeout_sec=timeout):
            req = SpawnEntity.Request()
            req.name = self.entity
            req.xml = str(entity_xml, 'utf-8')
            req.robot_namespace = ''
            req.initial_pose = initial_pose
            req.reference_frame = ''
            self.get_logger().info('Calling service /spawn_entity')
            srv_call = self.spawn_model.call_async(req)
            while rclpy.ok():
                if srv_call.done():
                    self.get_logger().info('Spawn status: %s' % srv_call.result().status_message)
                    break
                rclpy.spin_once(self)
            return srv_call.result().success
        self.get_logger().error(
            'Service %s/spawn_entity unavailable. Was Gazebo started with GazeboRosFactory?')
        return False

    def generate_poses(self,max_val=20.0, min_val=-20.0, max_diff=10.0):
        '''
        Generate random origin and goal poses within the given range and max_diff
        max_val: Maximum x and y value of the terrain available for spawning
        min_val: Minimum x and y value of the terrain available for spawning
        max_diff: Maximum distance between current pose and goal poses
        
        '''
        origin_x = np.random.uniform(min_val, max_val)
        origin_y = np.random.uniform(min_val, max_val)
        origin_yaw = self.quaternion_from_euler(0.0,0.0,np.random.uniform(0, 2 * np.pi) )  # Assuming yaw is in radians

        # Calculate the range for the grid
        x_min = max(origin_x - max_diff, min_val)
        x_max = min(origin_x + max_diff, max_val)
        y_min = max(origin_y - max_diff, min_val)
        y_max = min(origin_y + max_diff, max_val)

        # Generate grid of possible goal poses within max_diff
        x_range = np.arange(x_min, x_max, 0.1)
        y_range = np.arange(y_min, y_max, 0.1)
        goal_poses = [(x, y) for x in x_range for y in y_range if np.sqrt((x - origin_x)**2 + (y - origin_y)**2) <= max_diff]

        # Randomly select a goal pose from the list
        goal_x, goal_y = random.choice(goal_poses)
        goal_yaw = self.quaternion_from_euler(0.0,0.0,np.random.uniform(0, 2 * np.pi) )  # Assuming yaw is in radians

        origin_pose = Pose()
        origin_pose.position.x = origin_x
        origin_pose.position.y = origin_y
        origin_pose.position.z = 6.0      
        origin_pose.orientation.x = origin_yaw[0]
        origin_pose.orientation.y = origin_yaw[1]
        origin_pose.orientation.z = origin_yaw[2]
        origin_pose.orientation.w = origin_yaw[3]

        goal_pose = PoseStamped()
        goal_pose.pose.position.x = goal_x
        goal_pose.pose.position.y = goal_y
        goal_pose.pose.position.z = 0.8   
        goal_pose.pose.orientation.x = goal_yaw[0]
        goal_pose.pose.orientation.y = goal_yaw[1]
        goal_pose.pose.orientation.z = goal_yaw[2]
        goal_pose.pose.orientation.w = goal_yaw[3]

        print(f"[gazebo_connection] Respawn Pose: \n \
              position \nx:{origin_pose.position.x} \
                \ny:{origin_pose.position.y} \
                \nz:{origin_pose.position.z} \
                \n orintation: \nx{origin_pose.orientation.x} \
                \ny:{origin_pose.orientation.y} \
                \nz:{origin_pose.orientation.z} \
                \nw:{origin_pose.orientation.w} \
               \n\n Goal Pose:  \
               \n position \nx:{goal_pose.pose.position.x} \
                \ny:{goal_pose.pose.position.y} \
                \nz:{goal_pose.pose.position.z} \
                \n orintation: \nx{goal_pose.pose.orientation.x} \
                \ny:{goal_pose.pose.orientation.y} \
                \nz:{goal_pose.pose.orientation.z} \
                \nw:{goal_pose.pose.orientation.w}")


        return (origin_pose, goal_pose)






    def update_gravity_call(self):
        
        # self.pause_sim()

        # set_physics_request = SetPhysicsProperties_Request()
        # set_physics_request.time_step = self._time_step.data
        # set_physics_request.max_update_rate = self._max_update_rate.data
        # set_physics_request.gravity = self._gravity
        # set_physics_request.ode_config = self._ode_config

        # self.get_logger().info(str(set_physics_request.gravity))

        # future = self.set_physics.call_async(set_physics_request)
        # rclpy.spin_until_future_complete(self, future)
        # result = future.result()
        # self.get_logger().info("Gravity Update Result==" + str(result.success) + ", message==" + str(result.status_message))

        # self.unpause_sim()

        return NotImplementedError

    def change_gravity(self, x, y, z):
        # self._gravity.x = x
        # self._gravity.y = y
        # self._gravity.z = z

        # self.update_gravity_call()
        return NotImplementedError
    def delete_entity(self):
        '''
        Delete the spawned entity from gazebo
        '''

        # Delete entity from gazebo on shutdown if bond flag enabled
        self.get_logger().info(f'Deleting entity {self.entity} ') 
        if self.delete_model.wait_for_service(timeout_sec=self.timeout):
            req = DeleteEntity.Request()
            req.name = self.entity
            self.get_logger().info(
                'Calling service delete_entity')
            srv_call = self.delete_model.call_async(req)
            while rclpy.ok():
                if srv_call.done():
                    self.get_logger().info(
                        'Deleting status: %s' % srv_call.result().status_message)
                    break
                rclpy.spin_once(self)
        else:
            self.get_logger().error(
                'Service %s/delete_entity unavailable. ' +
                'Was Gazebo started with GazeboRosFactory?')

    
    def spawn_robot(self):
        '''
        Spawn the robot in the gazebo environment
        '''


        origin, goal = self.generate_poses(max_val=20.0, min_val=-20.0, max_diff=10.0)
        self.get_logger().info('Loading entity XML from file %s' % self.urdf)
        if not os.path.exists(self.urdf):
            self.get_logger().error('Error: specified file %s does not exist', self.urdf)
            return 1
        if not os.path.isfile(self.urdf):
            self.get_logger().error('Error: specified file %s is not a file', self.urdf)
            return 1
        # load file
        try:
            f = open(self.urdf, 'r')
            entity_xml = f.read()
        except IOError as e:
            self.get_logger().error('Error reading file {}: {}'.format(self.urdf, e))
            return 1
        if entity_xml == '':
            self.get_logger().error('Error: file %s is empty', self.urdf)
            return 1
        print(f"[gazebo_connection] After read     pose")
    
        try:
            xml_parsed = ElementTree.fromstring(entity_xml)
        except ElementTree.ParseError as e:
            self.get_logger().error('Invalid XML: {}'.format(e))
            return 1


        # Encode xml object back into string for service call
        entity_xml = ElementTree.tostring(xml_parsed)
        success=self._spawn_entity(entity_xml, origin,self.timeout)

        if success:
            
            self.goal_pose_pub.publish(goal)
            return 0
        else:
            self.get_logger().error('Spawn service failed. Exiting.')
            return 1


    
    def quaternion_from_euler(self,roll, pitch, yaw):
        '''
        Function to convert euler angles to quaternion
        '''
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        q = [0] * 4
        q[0] = cy * cp * cr + sy * sp * sr
        q[1] = cy * cp * sr - sy * sp * cr
        q[2] = sy * cp * sr + cy * sp * cr
        q[3] = sy * cp * cr - cy * sp * sr

        return q

# def main(args=None):
#     rclpy.logging._root_logger.log(
#         'Starting gazebo connection node ...',
#         LoggingSeverity.INFO
#     )
#     rclpy.init(args=args)
#     gazebo_connection = GazeboConnection()
#     rclpy.spin(gazebo_connection)
#     gazebo_connection.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()