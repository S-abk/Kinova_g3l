#! /usr/bin/env python3

###
# KINOVA (R) KORTEX (TM)
#
# Copyright (c) 2021 Kinova inc. All rights reserved.
#
# This software may be modified and distributed
# under the terms of the BSD 3-Clause license.
#
# Refer to the LICENSE file for details.
#
###

import sys
import os
import time
import threading

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient


from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2

# Maximum allowed waiting time during actions (in seconds)
TIMEOUT_DURATION = 30

# Create closure to set an event after an END or an ABORT
def check_for_end_or_abort(e):
    """Return a closure checking for END or ABORT notifications

    Arguments:
    e -- event to signal when the action is completed
        (will be set when an END or ABORT occurs)
    """
    def check(notification, e = e):
        print("EVENT : " + \
              Base_pb2.ActionEvent.Name(notification.action_event))
        if notification.action_event == Base_pb2.ACTION_END \
        or notification.action_event == Base_pb2.ACTION_ABORT:
            e.set()
    return check
 
def example_move_to_home_position(base):
    # Make sure the arm is in Single Level Servoing mode
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)
    
    # Move arm to ready position
    print("Moving the arm to a safe position")
    action_type = Base_pb2.RequestedActionType()
    action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
    action_list = base.ReadAllActions(action_type)
    action_handle = None
    for action in action_list.action_list:
        if action.name == "Home":
            action_handle = action.handle

    if action_handle == None:
        print("Can't reach safe position. Exiting")
        return False

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    base.ExecuteActionFromReference(action_handle)
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if finished:
        print("Safe position reached")
    else:
        print("Timeout on action notification wait")
    return finished
def populateCartesianCoordinate(waypointInformation):
    
    waypoint = Base_pb2.CartesianWaypoint()  
    waypoint.pose.x = waypointInformation[0]
    waypoint.pose.y = waypointInformation[1]
    waypoint.pose.z = waypointInformation[2]
    waypoint.blending_radius = waypointInformation[3]
    waypoint.pose.theta_x = waypointInformation[4]
    waypoint.pose.theta_y = waypointInformation[5]
    waypoint.pose.theta_z = waypointInformation[6] 
    waypoint.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
    
    return waypoint

def example_trajectory(base, base_cyclic):

    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)
    product = base.GetProductConfiguration()
    waypointsDefinition = tuple(tuple())
    if(   product.model == Base_pb2.ProductConfiguration__pb2.MODEL_ID_L53 
       or product.model == Base_pb2.ProductConfiguration__pb2.MODEL_ID_L31):
        if(product.model == Base_pb2.ProductConfiguration__pb2.MODEL_ID_L31):
            kTheta_x = 90.6
            kTheta_y = -1.0
            kTheta_z = 150.0
            waypointsDefinition = ( (0.439,  0.194, 0.448, 0.0, kTheta_x, kTheta_y, kTheta_z),
                                    (0.200,  0.150, 0.400, 0.0, kTheta_x, kTheta_y, kTheta_z),
                                    (0.350,  0.050, 0.300, 0.0, kTheta_x, kTheta_y, kTheta_z))
        else:
            kTheta_x = 90.0
            kTheta_y = 0.0
            kTheta_z = 90.0
            waypointsDefinition = ( (0.7,   0.0,  0.5,  0.0, kTheta_x, kTheta_y, kTheta_z),
                                    (0.7,   0.0,  0.33, 0.1, kTheta_x, kTheta_y, kTheta_z),
                                    (0.7,   0.48, 0.33, 0.1, kTheta_x, kTheta_y, kTheta_z),
                                    (0.61,  0.22, 0.4,  0.1, kTheta_x, kTheta_y, kTheta_z),
                                    (0.7,   0.48, 0.33, 0.1, kTheta_x, kTheta_y, kTheta_z),
                                    (0.63, -0.22, 0.45, 0.1, kTheta_x, kTheta_y, kTheta_z),
                                    (0.65,  0.05, 0.33, 0.0, kTheta_x, kTheta_y, kTheta_z))
    else:
        print("Product is not compatible to run this example please contact support with KIN number bellow")
        print("Product KIN is : " + product.kin())

    
    waypoints = Base_pb2.WaypointList()
    
    waypoints.duration = 0.0
    waypoints.use_optimal_blending = False
    
    index = 0
    for waypointDefinition in waypointsDefinition:
        waypoint = waypoints.waypoints.add()
        waypoint.name = "waypoint_" + str(index)   
        waypoint.cartesian_waypoint.CopyFrom(populateCartesianCoordinate(waypointDefinition))
        index = index + 1 

    # Verify validity of waypoints
    result = base.ValidateWaypointList(waypoints);
    if(len(result.trajectory_error_report.trajectory_error_elements) == 0):
        e = threading.Event()
        notification_handle = base.OnNotificationActionTopic(   check_for_end_or_abort(e),
                                                                Base_pb2.NotificationOptions())

        print("Moving cartesian trajectory...")
        
        base.ExecuteWaypointTrajectory(waypoints)

        print("Waiting for trajectory to finish ...")
        finished = e.wait(TIMEOUT_DURATION)
        base.Unsubscribe(notification_handle)

        if finished:
            print("Cartesian trajectory with no optimization completed ")
            e_opt = threading.Event()
            notification_handle_opt = base.OnNotificationActionTopic(   check_for_end_or_abort(e_opt),
                                                                Base_pb2.NotificationOptions())

            waypoints.use_optimal_blending = True
            base.ExecuteWaypointTrajectory(waypoints)

            print("Waiting for trajectory to finish ...")
            finished_opt = e_opt.wait(TIMEOUT_DURATION)
            base.Unsubscribe(notification_handle_opt)

            if(finished_opt):
                print("Cartesian trajectory with optimization completed ")
            else:
                print("Timeout on action notification wait for optimized trajectory")

            return finished_opt
        else:
            print("Timeout on action notification wait for non-optimized trajectory")

        return finished
        
    else:
        print("Error found in trajectory") 
        result.trajectory_error_report.PrintDebugString();  


def example_circular_trajectory(base, base_cyclic):
    import math
    # Set the robot to Single Level Servoing mode
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)

    # Define parameters for the circular trajectory
    center_x = 0.5    # Center x-coordinate (in meters)
    center_y = 0.0    # Center y-coordinate (in meters)
    center_z = 0.4    # Constant height (in meters)
    radius = 0.15     # Radius of the circle (in meters)
    num_points = 12   # Number of waypoints (more than 9 for a smooth path)
    blending_radius = 0.05  # Blending radius for smooth transitions

    # Orientation (assumed constant for all waypoints)
    theta_x = 90.0
    theta_y = 0.0
    theta_z = 90.0

    # Build the list of waypoints along a circle
    waypointsDefinition = []
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        z = center_z
        waypointsDefinition.append((x, y, z, blending_radius, theta_x, theta_y, theta_z))
    
    # Create the waypoint list message
    waypoints = Base_pb2.WaypointList()
    waypoints.duration = 0.0  # Immediate execution, timing is handled by the trajectory
    waypoints.use_optimal_blending = True  # Enable smooth transitions

    for index, wp_def in enumerate(waypointsDefinition):
        waypoint = waypoints.waypoints.add()
        waypoint.name = f"circle_wp_{index}"
        waypoint.cartesian_waypoint.CopyFrom(populateCartesianCoordinate(wp_def))

    # Validate the waypoint list before execution
    result = base.ValidateWaypointList(waypoints)
    if len(result.trajectory_error_report.trajectory_error_elements) == 0:
        e = threading.Event()
        notification_handle = base.OnNotificationActionTopic(
            check_for_end_or_abort(e),
            Base_pb2.NotificationOptions()
        )

        print("Executing circular trajectory...")
        base.ExecuteWaypointTrajectory(waypoints)
        finished = e.wait(TIMEOUT_DURATION)
        base.Unsubscribe(notification_handle)

        if finished:
            print("Circular trajectory completed successfully.")
        else:
            print("Timeout during circular trajectory execution.")
        return finished
    else:
        print("Trajectory validation failed:")
        result.trajectory_error_report.PrintDebugString()
        return False


def main():
    # Import the utilities helper module
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import utilities

    # Parse arguments
    args = utilities.parseConnectionArguments()
    
    # Create connection to the device and get the router
    with utilities.DeviceConnection.createTcpConnection(args) as router:
        # Create required services
        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)
        
        # Example core
        success = True

        # Move to a safe home position
        success &= example_move_to_home_position(base)
        
        # Execute the original trajectory demo (if desired)
        success &= example_trajectory(base, base_cyclic)
        
        # Execute the new circular trajectory demo
        success &= example_circular_trajectory(base, base_cyclic)
       
        return 0 if success else 1

if __name__ == "__main__":
    exit(main())
