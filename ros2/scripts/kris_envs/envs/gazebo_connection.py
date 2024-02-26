#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty
# from gazebo_msgs.msg import ODEPhysics
from gazebo_msgs.srv import SetPhysicsProperties, SpawnEntity, DeleteEntity


class GazeboConnection(Node):

    def __init__(self):
        super().__init__('gazebo_connection_node')

        self.unpause = self.create_client(Empty, '/unpause_physics')
        self.pause = self.create_client(Empty, '/pause_physics')
        self.reset_proxy = self.create_client(Empty, '/reset_simulation')
        self.spawn_model = self.create_client(SpawnEntity, '/spawn_entity')
        self.delete_model = self.create_client(DeleteEntity, '/delete_entity')

        # Setup the Gravity Control system
        service_name = '/gazebo/set_parameters'
        self.set_physics = self.create_client(SetPhysicsProperties, service_name)
        self.get_logger().info("Waiting for service " + str(service_name))
        while not self.set_physics.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Service not available, waiting again...")
        self.get_logger().info("Service Found " + str(service_name))

        self.init_values()
        # We always pause the simulation, important for legged robots learning
        self.pause_sim()

    def pause_sim(self):
        request = Empty.Request()
        future = self.pause.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info("/gazebo/pause_physics service call successful")
        else:
            self.get_logger().error("/gazebo/pause_physics service call failed")

    def unpause_sim(self):
        request = Empty.Request()
        future = self.unpause.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info("/gazebo/unpause_physics service call successful")
        else:
            self.get_logger().error("/gazebo/unpause_physics service call failed")

    def reset_sim(self):
        request = Empty.Request()
        future = self.reset_proxy.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info("/gazebo/reset_simulation service call successful")
        else:
            self.get_logger().error("/gazebo/reset_simulation service call failed")

    def reset_world(self):
        self.reset_sim()  # Assuming reset_world is equivalent to reset_simulation in ROS2
        self.respwan_robot()

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
    
    def respwan_robot(self):
        return NotImplementedError


def main(args=None):
    rclpy.init(args=args)
    gazebo_connection = GazeboConnection()
    rclpy.spin(gazebo_connection)
    gazebo_connection.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
