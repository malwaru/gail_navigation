#!/usr/bin/env python

import rclpy
from std_srvs.srv import Empty
from gazebo_msgs.msg import ODEPhysics
from gazebo_msgs.srv import SetPhysicsProperties, SetPhysicsPropertiesRequest
from std_msgs.msg import Float64
from geometry_msgs.msg import Vector3

class GazeboConnection():
    
    def __init__(self):
        
        self.unpause = rclpy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rclpy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rclpy.ServiceProxy('/gazebo/reset_simulation', Empty)

        # Setup the Gravity Controle system
        service_name = '/gazebo/set_physics_properties'
        rclpy.logdebug("Waiting for service " + str(service_name))
        rclpy.wait_for_service(service_name)
        rclpy.logdebug("Service Found " + str(service_name))

        self.set_physics = rclpy.ServiceProxy(service_name, SetPhysicsProperties)
        self.init_values()
        # We always pause the simulation, important for legged robots learning
        self.pauseSim()

    def pauseSim(self):
        rclpy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except rclpy.ServiceException as e:
            print ("/gazebo/pause_physics service call failed")
        
    def unpauseSim(self):
        rclpy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rclpy.ServiceException as e:
            print ("/gazebo/unpause_physics service call failed")
        
    def resetSim(self):
        rclpy.wait_for_service('/gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except rclpy.ServiceException as e:
            print ("/gazebo/reset_simulation service call failed")

    def resetWorld(self):
        rclpy.wait_for_service('/gazebo/reset_world')
        try:
            self.reset_proxy()
        except rclpy.ServiceException as e:
            print ("/gazebo/reset_world service call failed")

    def init_values(self):

        rclpy.wait_for_service('/gazebo/reset_simulation')
        try:
            # reset_proxy.call()
            self.reset_proxy()
        except rclpy.ServiceException as e:
            print ("/gazebo/reset_simulation service call failed")

        self._time_step = Float64(0.001)
        self._max_update_rate = Float64(1000.0)
        

        self._gravity = Vector3()
        self._gravity.x = 0.0
        self._gravity.y = 0.0
        self._gravity.z = 0.0

        self._ode_config = ODEPhysics()
        self._ode_config.auto_disable_bodies = False
        self._ode_config.sor_pgs_precon_iters = 0
        self._ode_config.sor_pgs_iters = 50
        self._ode_config.sor_pgs_w = 1.3
        self._ode_config.sor_pgs_rms_error_tol = 0.0
        self._ode_config.contact_surface_layer = 0.001
        self._ode_config.contact_max_correcting_vel = 0.0
        self._ode_config.cfm = 0.0
        self._ode_config.erp = 0.2
        self._ode_config.max_contacts = 20

        self.update_gravity_call()

    def update_gravity_call(self):

        self.pauseSim()

        set_physics_request = SetPhysicsPropertiesRequest()
        set_physics_request.time_step = self._time_step.data
        set_physics_request.max_update_rate = self._max_update_rate.data
        set_physics_request.gravity = self._gravity
        set_physics_request.ode_config = self._ode_config

        rclpy.logdebug(str(set_physics_request.gravity))

        result = self.set_physics(set_physics_request)
        rclpy.logdebug("Gravity Update Result==" + str(result.success) + ",message==" + str(result.status_message))

        self.unpauseSim()

    def change_gravity(self, x, y, z):
        self._gravity.x = x
        self._gravity.y = y
        self._gravity.z = z

        self.update_gravity_call()