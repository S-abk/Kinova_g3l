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



def example_manual_trajectory(base, base_cyclic):
    """
    This function allows you to manually define and test a trajectory by inputting waypoints.
    
    Each waypoint is defined as a tuple:
        (x, y, z, blending_radius, theta_x, theta_y, theta_z)
        
    Notes:
    - Coordinates are in meters.
    - Orientations are in degrees.
    - The final waypoint should have a blending radius of 0.0 (no blending).
    - The trajectory is executed in the robot's base reference frame.
    """
    # Set the robot to Single Level Servoing mode
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)

    # Editable list of manual waypoints.
    # Modify these tuples to test different trajectories.
    kTheta_x = 10.772
    kTheta_y = 177.866
    kTheta_z = 82.777
    manual_waypoints = [
        (0.3,  -0.104,  0.0, 0.0, kTheta_x, kTheta_y, kTheta_z),
        (0.291, -0.104,  0.2, 0.1, kTheta_x, kTheta_y, kTheta_z),
        (0.291, 0.150,  0.2,  0.1, kTheta_x, kTheta_y, kTheta_z),
        (0.291, 0.150,  0.0, 0.1, kTheta_x, kTheta_y, kTheta_z),
        (0.291, 0.0,  0.3,  0.1, kTheta_x, kTheta_y, kTheta_z),
        (0.291, 0.0,  0.3, 0.0, kTheta_x, kTheta_y, kTheta_z) # Final waypoint: blending_radius must be 0.0
    ]
    
    # Create the waypoint list message
    waypoints = Base_pb2.WaypointList()
    waypoints.duration = 0.0  # Duration not explicitly set (immediate execution) me
    waypoints.use_optimal_blending = True  # Enable smooth transitions

    # Populate the waypoint list using the manual inputs
    for index, wp_def in enumerate(manual_waypoints):
        waypoint = waypoints.waypoints.add()
        waypoint.name = f"manual_wp_{index}"
        waypoint.cartesian_waypoint.CopyFrom(populateCartesianCoordinate(wp_def))

    # Validate the waypoint list
    result = base.ValidateWaypointList(waypoints)
    if len(result.trajectory_error_report.trajectory_error_elements) == 0:
        e = threading.Event()
        notification_handle = base.OnNotificationActionTopic(
            check_for_end_or_abort(e),
            Base_pb2.NotificationOptions()
        )

        print("Executing manual trajectory...")
        base.ExecuteWaypointTrajectory(waypoints)
        finished = e.wait(TIMEOUT_DURATION)
        base.Unsubscribe(notification_handle)

        if finished:
            print("Manual trajectory completed successfully.")
        else:
            print("Timeout during manual trajectory execution.")
        return finished
    else:
        print("Trajectory validation failed:")
        for error in result.trajectory_error_report.trajectory_error_elements:
            print(f"Error: {error}")
        return False





def open_gripper(base):
    """
    Opens the gripper by sending a position command (1.0 for fully open).
    """
    gripper_command = Base_pb2.GripperCommand()
    finger = gripper_command.gripper.finger.add()
    gripper_command.mode = Base_pb2.GRIPPER_POSITION
    finger.finger_identifier = 1
    finger.value = 0.0  # Fully open
    base.SendGripperCommand(gripper_command)
    print("Gripper opened.")

def close_gripper(base):
    """
    Closes the gripper by sending a position command (0.0 for fully closed).
    """
    gripper_command = Base_pb2.GripperCommand()
    finger = gripper_command.gripper.finger.add()
    gripper_command.mode = Base_pb2.GRIPPER_POSITION
    finger.finger_identifier = 1
    finger.value = 0.6  # 60% closed
    base.SendGripperCommand(gripper_command)
    print("Gripper closed.")

def pick_and_place_demo(base, base_cyclic):
    """
    Executes a pick and place operation by combining trajectory motions with gripper commands.
    The operation includes:
      - Moving to a pre-pick position.
      - Opening the gripper.
      - Lowering to pick an object.
      - Closing the gripper.
      - Lifting the object.
      - Moving to a pre-place position.
      - Lowering to place the object.
      - Opening the gripper to release.
      - Retracting to a safe position.
      
    All positions are defined in the robot's base reference frame.
    """
    import time

    # Set the robot to Single Level Servoing mode.
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)

    # Common orientation (in degrees) and blending values.
    theta_x = 10.772
    theta_y = 177.866
    theta_z = 82.777
    blending = 0.05

    # Define key waypoints (x, y, z, blending_radius, theta_x, theta_y, theta_z)
    pre_pick  = (0.3, -0.1, 0.1, 0.0, theta_x, theta_y, theta_z)  # Approach above object
    pick      = (0.3, -0.1, 0.0, blending, theta_x, theta_y, theta_z)  # Lower to object
    post_pick = (0.3, -0.1, 0.1, blending, theta_x, theta_y, theta_z)  # Lift object
    pre_place = (0.3, 0.1, 0.1, blending, theta_x, theta_y, theta_z)  # Move over destination
    place     = (0.3, 0.1, 0.0, blending, theta_x, theta_y, theta_z)  # Lower to place
    post_place= (0.3, 0.1, 0.1, 0.0,    theta_x, theta_y, theta_z)  # Retract (final waypoint with blending=0)

    # Helper function to execute a simple trajectory from the current pose to a target waypoint.
    def execute_trajectory(waypoints_list):
        waypoints = Base_pb2.WaypointList()
        waypoints.duration = 0.0  # Timing controlled by the trajectory
        waypoints.use_optimal_blending = True
        for idx, wp in enumerate(waypoints_list):
            waypoint = waypoints.waypoints.add()
            # Ensure the final waypoint has zero blending radius.
            if idx == len(waypoints_list) - 1:
                wp = (wp[0], wp[1], wp[2], 0.0, wp[4], wp[5], wp[6])
            waypoint.name = f"pp_wp_{idx}"
            waypoint.cartesian_waypoint.CopyFrom(populateCartesianCoordinate(wp))
        
        # Validate the trajectory.
        result = base.ValidateWaypointList(waypoints)
        if len(result.trajectory_error_report.trajectory_error_elements) == 0:
            e = threading.Event()
            notification_handle = base.OnNotificationActionTopic(
                check_for_end_or_abort(e),
                Base_pb2.NotificationOptions()
            )
            base.ExecuteWaypointTrajectory(waypoints)
            finished = e.wait(TIMEOUT_DURATION)
            base.Unsubscribe(notification_handle)
            return finished
        else:
            print("Trajectory validation failed:")
            for error in result.trajectory_error_report.trajectory_error_elements:
                print(f"Error: {error}")
            return False

    # Execute the sequence of motions with delays for safety.
    print("Starting pick and place operation...")

    # 1. Move to pre-pick position.
    print("Moving to pre-pick position...")
    if not execute_trajectory([pre_pick]):
        return False
    time.sleep(1)

    # 2. Open the gripper.
    print("Opening gripper...")
    open_gripper(base)
    time.sleep(1)

    # 3. Lower to pick position.
    print("Lowering to pick position...")
    if not execute_trajectory([pick]):
        return False
    time.sleep(1)

    # 4. Close the gripper to grasp the object.
    print("Closing gripper to grasp object...")
    close_gripper(base)
    time.sleep(2) # allow the gripper to fully close

    # 5. Lift the object (move back to post-pick position).
    print("Lifting object...")
    if not execute_trajectory([post_pick]):
        return False
    time.sleep(1)

    # 6. Move to pre-place (over the destination).
    print("Moving to pre-place position...")
    if not execute_trajectory([pre_place]):
        return False
    time.sleep(1)

    # 7. Lower to place position.
    print("Lowering to place position...")
    if not execute_trajectory([place]):
        return False
    time.sleep(1)

    # 8. Open the gripper to release the object.
    print("Releasing object by opening gripper...")
    open_gripper(base)
    time.sleep(1)

    # 9. Retract to post-place position.
    print("Retracting after placing object...")
    if not execute_trajectory([post_place]):
        return False
    time.sleep(1)

    print("Pick and place operation completed successfully.")
    return True


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


        # Execute the manual trajectory demo
        # success &= manual_trajectory(base, base_cyclic)

        # Execute the manual trajectory demo
        # success &= example_manual_trajectory(base, base_cyclic)
        
        # Execute the original trajectory demo (if desired)
        # success &= example_trajectory(base, base_cyclic)
        


        # Execute the pick and place demo
        success &= pick_and_place_demo(base, base_cyclic)


        # Move to a safe home position
        # success &= example_move_to_home_position(base)



       
        return 0 if success else 1

if __name__ == "__main__":
    exit(main())
