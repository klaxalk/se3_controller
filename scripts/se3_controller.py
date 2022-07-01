#!/usr/bin/python3

import rospy
import rosnode
import math
import numpy as np

from pyquaternion import Quaternion

from mrs_msgs.srv import ActuatorControl as ActuatorControl
from mrs_msgs.srv import ActuatorControlResponse as ActuatorControlResponse
from mrs_msgs.srv import ActuatorControlRequest as ActuatorControlRequest

class ActuatorController:

    def actuatorControlCallback(self, data):

        gain_pos = self.gain_pos * self.mass
        gain_vel = self.gain_vel * self.mass
        gain_rot = self.gain_rot * self.mass
        gain_rate = self.gain_rate * self.mass

        # position control error
        err_pos = np.array([data.reference.position.x - data.uav_state.pose.position.x,
                            data.reference.position.y - data.uav_state.pose.position.y,
                            data.reference.position.z - data.uav_state.pose.position.z])

        # velocity control error
        err_vel = np.array([data.reference.velocity.x - data.uav_state.velocity.linear.x,
                            data.reference.velocity.y - data.uav_state.velocity.linear.y,
                            data.reference.velocity.z - data.uav_state.velocity.linear.z])

        # gravity compensation vector
        gravity_comp = np.array([0,
                                 0,
                                 self.mass * self.g]).transpose()

        # desired force
        des_F = gain_pos * err_pos + gain_vel * err_vel + gravity_comp
        des_F_normalized = des_F / np.linalg.norm(des_F)

        # desired heading vec
        des_heading_vec = np.array([math.cos(data.reference.heading),
                                    math.sin(data.reference.heading),
                                    0])

        # desired orientation matrix
        des_R = np.eye(3)

        des_R[:, 2] = des_F_normalized
        des_R[:, 1] = np.cross(des_R[:, 2], des_heading_vec)
        des_R[:, 1] = des_R[:, 1] / np.linalg.norm(des_R[:, 1])
        des_R[:, 0] = np.cross(des_R[:, 1], des_R[:, 2])
        des_R[:, 0] = des_R[:, 0] / np.linalg.norm(des_R[:, 0])

        # orientation error
        orient_quat = Quaternion(data.uav_state.pose.orientation.w, data.uav_state.pose.orientation.x, data.uav_state.pose.orientation.y, data.uav_state.pose.orientation.z)
        R = orient_quat.rotation_matrix

        err_R = 0.5 * (des_R.transpose().dot(R) - R.transpose().dot(des_R))
        err_R_vec = -np.array([0.5 * (err_R[2, 1] - err_R[1, 2]),
                               0.5 * (err_R[0, 2] - err_R[2, 0]),
                               0.5 * (err_R[1, 0] - err_R[0, 1])]) 

        # desired angular rate
        des_rate = gain_rot * err_R_vec

        # rate error
        err_rate = np.array([des_rate[0] - data.uav_state.velocity.angular.x,
                             des_rate[1] - data.uav_state.velocity.angular.y,
                             des_rate[2] - data.uav_state.velocity.angular.z])

        # desired thrust
        des_F_scalar = des_F.dot(R[:, 2])
        des_thrust = math.sqrt(des_F_scalar / 4.0) * self.motor_param_A + self.motor_param_B

        # control group commands
        controls = np.array([gain_rate * err_rate[0],
                             -gain_rate * err_rate[1],
                             -gain_rate * err_rate[2],
                             des_thrust])

        # allocation matrix
        mixer = np.array([[-0.707107,  0.707107,  1.000000,  1.000000],
                          [0.707107, -0.707107,  1.000000,   1.000000],
                          [ 0.707107,  0.707107, -1.000000,  1.000000],
                          [-0.707107, -0.707107, -1.000000,  1.000000]])

        # motor commands
        motors = mixer.dot(controls)

        response = ActuatorControlResponse()

        response.motors[0] = motors[0]
        response.motors[1] = motors[1]
        response.motors[2] = motors[2]
        response.motors[3] = motors[3]

        rospy.loginfo('outputting motor commands: {} {} {} {}'.format(motors[0], motors[1], motors[2], motors[3]))

        response.success = True

        return response

    def __init__(self):

        rospy.init_node('actuator_controller', anonymous=True)

        self.gain_pos = rospy.get_param("~gains/position")
        self.gain_vel = rospy.get_param("~gains/velocity")
        self.gain_rot = rospy.get_param("~gains/rotation")
        self.gain_rate = rospy.get_param("~gains/rate")

        self.mass = rospy.get_param("~mass")
        self.g = rospy.get_param("~g")
        self.motor_param_A = rospy.get_param("~motor_params/a")
        self.motor_param_B = rospy.get_param("~motor_params/b")

        self.service_server = rospy.Service("~actuator_control_srv_in", ActuatorControl, self.actuatorControlCallback)

        rospy.loginfo('initialized')

        rospy.spin()

if __name__ == '__main__':
    try:

        actuator_controller = ActuatorController()

    except rospy.ROSInterruptException:
        pass
