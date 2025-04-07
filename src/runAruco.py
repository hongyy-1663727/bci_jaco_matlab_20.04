#! /usr/bin/env python3

import rospy
import numpy as np
import os
import interfacesAruco
import pandas as pd
from sensor_msgs.msg import Image
from std_msgs.msg import UInt8MultiArray
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
import tf.transformations

# Initialize ROS node
bridge = CvBridge()

interface = interfacesAruco.DiscreteActionsRobot()
interface.reset()
interface.aruco_detected = False
interface.target_pos = [0.0, 0.0, 0.0]  # Default position


# Process camera images
def image_callback(msg):
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        interface.processImage(cv_image)  # This processes and displays the image
    except Exception as e:
        rospy.logerr("Error processing image: %s", str(e))

# Process commands (replace UDP with ROS messages)
def command_callback(msg):
    # Process commands similar to how they were handled via UDP
    data = msg.data
    command = data[0]
    val1 = data[1]
    val2 = data[2]
    val3 = data[3]
    val4 = data[4]
    val5 = data[5]
    val6 = data[6]
    val7 = data[7]
    val8 = data[8]
    val9 = data[9]
    val10 = data[10]

    if command == 0:
        if val1 == 1:		# Reset 
            interface.reset()
            print("RESET")
        if val1 == 2:		# Set Mode
            interface.setMode(val2)
            print("Mode: ", val2)
        if val1 == 3:		# Set 
            interface.targetRadius = val2/1000.0
        if val1 == 4:		# Set 
            # interface.targetHoldTime = val2
            interface.holdTimeSteps  = val2/0.2
            # interface.holdTimeSteps = 1
            print(interface.holdTimeSteps)

        if val1 == 6:
            H = val2;
            M = val3;
            S = val4;
            dataDir = (
                f"Data/Bravo1/{time.strftime('%Y%m%d')}/"
                f"{str(H).zfill(2)}{str(M).zfill(2)}{str(S).zfill(2)}"
            )
            if not os.path.exists(dataDir):
                os.makedirs(dataDir)

        if val1 == 5:  #Start logger
            trialInd = len(os.listdir(dataDir))
            print(trialInd)
            fn = f"{dataDir}/data00{trialInd}.csv"
            interface.startLogger(fn)
            interface.inTrial = 1

        if val1 == 7:		# Set Alpha
            interface.assistAlpha = val2/100.0
            print("Alpha ", val2/100.0)

        if val1 == 8:		# Set AutoGrasp
            interface.AG = val2

        if val1 == 9:		# Set clicker hold time
            interface.runningSwitchBinNum  = val2

        if val1 == 10:		# Set autocenter flag
            interface.autoCenterOverTarget  = val2

        if val1 == 11:		# Set autocenter dist
            interface.autoCenterDist  = val2*.01

        if val1 == 12:		# Set wrist start X
            interface.wristStartX  = val2*.1
            print(interface.wristStartX)

        if val1 == 13:		# Set wrist start Z
            interface.wristStartZ  = val2*.1

        if val1 == 14:		# Set operational mode reset
            interface.operationalModeReset = val2

        if val1 == 15:		# Set WL[2]
            interface.wl[2] = val2*.01

        if val1 == 16:		# Set low gain mode
            interface.lowGainMode = val2
            
        if val1 == 17:		# Set grasp orientation
            interface.graspOrientation = val2
        if val1 == 18:		# 
            interface.runningSwitchBinNum = val2
        if val1 == 19:		# 
            interface.runningSwitchThresh = val2*0.1
        if val1 == 20:		# 
            interface.runningGraspBinNum = val2
        if val1 == 21:		# 
            interface.runningGraspThresh = val2*0.1
        if val1 == 41:
            interface.ArucoAssist = val2
            

        if val1 == 22:
            interface.wl[0] = ((val2-1) *(val3 + val4/100.0))/ 100.0
            interface.wl[1] = ((val5-1) *(val6 + val7/100.0))/ 100.0
            interface.wl[2] = ((val8-1) *(val9 + val10/100.0))/ 100.0
            print("WL", interface.wl )
        if val1 == 23:			
            interface.wu[0] = ((val2-1) *(val3 + val4/100.0))/ 100.0
            interface.wu[1] = ((val5-1) *(val6 + val7/100.0))/ 100.0
            interface.wu[2] = ((val8-1) *(val9 + val10/100.0))/ 100.0
            print("WU", interface.wu )
        if val1 == 25:		# Stop for Beta Stop Task
            pose_str = str(interface.pose[1]).encode()
            sock.sendto(pose_str, ("127.0.0.1", 43210))
            interface.stopRobot()
        if val1 == 26:		# Set robot dynamics
            interface.kv_f 	= val2/10.0
            interface.ki_f 	= val3/1000.0
            interface.gkv 	= val4/10.0
            interface.gki 	= val5/1000.0
            print("DYN:", interface.kv_f , interface.ki_f , interface.gkv ,interface.gki )
        if val1 == 27:		
            if val2 == 1:
                print("CLOSE")
                interface.setGripper(interface.gripperClose)
            if val2 == 2:
                interface.setGripper(interface.gripperOpen)

        if val1 == 28: # set next path goal
            interface.goalVal[1] = -((val2-1) *(val3 + val4/100.0))/ 1000.0 - 0.4
            interface.goalVal[0] = -((val5-1) *(val6 + val7/100.0))/ 1000.0 - 0.2
            interface.goalVal[2] = ((val8-1) *(val9 + val10/100.0) + 256)/ 1000.0

        if val1 == 29: # set goal Decode
            interface.goalDim = val2
            interface.goalMet = 0
            # update goal text
            interface.userDisplay.updateText(val2)

        if val1 == 30:
            print("CHANGE, val2")
            if val2 == 1:
                interface.userDisplay.changeBGColor((255,0,0))
            elif val2 == 0:
                interface.userDisplay.changeBGColor((0,255,0))

        if val1 == 31:		# Set Wrist Orientation
            interface.wristStartX  = ((val2-1) *(val3 + val4/100.0))/ 10.0
            interface.wristStartY  = ((val5-1) *(val6 + val7/100.0))/ 10.0
            interface.wristStartZ  = ((val8-1) *(val9 + val10/100.0))/ 10.0

        if val1 == 32:  #flipping detected
            print('FLIP STOP')
            interface.flipStop()

        if val1 == 33:
            interface.UseAutoGraspTD 		= val2
            interface.WaitForGraspSignal 	= val3
            interface.UseHeightOffset 		= val4
            interface.AutoGraspHorzDist     = val5/100.0
            interface.AutoGraspVertDist 	= val6/100.0

            if interface.UseHeightOffset == 0:
                interface.AutoGraspVertDist = 0

        if val1 == 34:  # Read goal pos from file
            vision_df = pd.read_csv('/home/sarah/Downloads/BCI_vision/eye_on_base_data.csv')
            print(vision_df)
            if len(vision_df) > 0:
                min_real_dis_row = vision_df.iloc[0]
                x = float(min_real_dis_row['real_x'].strip(']').strip('['))
                z = float(min_real_dis_row['real_y'].strip(']').strip('['))
                y = -float(min_real_dis_row['real_dis'].strip(']').strip('['))

                print(x,y,z)

                interface.t1_pos0[0] = x + 0.02
                interface.t1_pos0[1] = y + 0.1
                interface.t1_pos0[2] = z+ 0.02


                interface.t1_gpos0[0] = x + 0.02
                interface.t1_gpos0[1] = y - 0.
                interface.t1_gpos0[2] = z+ 0.02

                if len(vision_df) > 1:
                    min_real_dis_row = vision_df.iloc[1]
                    x = float(min_real_dis_row['real_x'].strip(']').strip('['))
                    z = float(min_real_dis_row['real_y'].strip(']').strip('['))
                    y = -float(min_real_dis_row['real_dis'].strip(']').strip('['))

                    print(x,y,z)

                    interface.t1_pos1[0] = x+ 0.02
                    interface.t1_pos1[1] = y + 0.1
                    interface.t1_pos1[2] = z + 0.02

                    interface.t1_gpos1[0] = x + 0.02
                    interface.t1_gpos1[1] = y - 0.
                    interface.t1_gpos1[2] = z + 0.02

        if val1 == 36:		# 
            interface.view = val2

        if val1 == 40: # set bg color
            if val2 == 0:
                interface.userDisplay.changeBGColor((0,0,0))
            if val2 == 1:
                interface.userDisplay.changeBGColor((255,0,0))
            if val2 == 2:
                interface.userDisplay.changeBGColor((0,255,0))
            if val2 == 3:
                interface.userDisplay.changeBGColor((0,0,255))


    # if command == 1:	# Read target position
    # 	interface.targetDone = 0
    # 	interface.target_pos = target_pos + [-0.20, -0.40, 0]
    # 	interface.TargetID = val10
    # 	interface.t1.pos = np.array(interface.target_pos)

    if command == 2:	# Read target1Pos
        target_pos[1] = -((val1-1) *(val2 + val3/100.0))/ 1000.0
        target_pos[0] = -((val4-1) *(val5 + val6/100.0))/ 1000.0
        target_pos[2] = ((val7-1) *(val8 + val9/100.0) + 256)/ 1000.0
        interface.t1_pos = target_pos + np.array([-0.20, -0.40, 0.0])
        print('targetPos: ', interface.t1_pos )

    if command == 3:	# Read target2Pos
        target_pos[1] = -((val1-1) *(val2 + val3/100.0))/ 1000.0
        target_pos[0] = -((val4-1) *(val5 + val6/100.0))/ 1000.0
        target_pos[2] = ((val7-1) *(val8 + val9/100.0) + 256)/ 1000.0
        interface.t2_pos = target_pos + np.array([-0.20, -0.40, 0.0])
        print('targetPos: ', interface.t2_pos)

    # if command == 3:	# Read targetPos
    # 	target_ang = -((val1-1) *(val2 + val3/100.0))/ 1000.0
    # 	interface.target_ang = target_ang

    if command == 4:	# Read position goal 
        robot_pos[1] = -((val1-1) *(val2 + val3/100.0))/ 1000.0
        robot_pos[0] = -((val4-1) *(val5 + val6/100.0))/ 1000.0
        robot_pos[2] = ((val7-1) *(val8 + val9/100.0) + 256)/ 1000.0
        interface.matlabPos = robot_pos
        interface.home = [interface.matlabPos[0] - 0.2, interface.matlabPos[1] - .4, interface.matlabPos[2]]
        interface.reset()

        print(robot_pos)

    if command == 5:	# Read Decoder Output
        interface.inputToAction(val10)
        interface.setVelocity(interface.V, interface.R)
        modeswitch_str = str(interface.modeswitch).encode()
        sock.sendto(modeswitch_str, ("127.0.0.1", 43210))

    if command == 6:	# Read Decoder Output (Path Task)
        # if interface.inTrial:
        print(interface.pose)
        interface.inputToAction(val10)
        interface.setVelocity(interface.V, interface.R)

        interface.checkPathGoal()
        goalMet_str = str(interface.goalMet).encode()
        sock.sendto(goalMet_str, ("127.0.0.1", 43210))
        if interface.goalMet:
            interface.V = [0,0,0]
            interface.R = [0,0,0]

    if command == 7:	# Read Decoder Output (Beta Stop Task)
        # if interface.inTrial:
        pose_str = str(interface.pose[1]).encode()
        sock.sendto(pose_str, ("127.0.0.1", 43210))
        interface.inputToAction(val10)
        interface.setVelocity(interface.V, interface.R)


def timer_callback(event):
    """Display ArUco detection status periodically"""
    if hasattr(interface, 'aruco_detected') and interface.aruco_detected:
        if hasattr(interface, 'aruco_timestamp') and (rospy.Time.now() - interface.aruco_timestamp).to_sec() < 1.0:
            print(f"ArUco marker detected at: {np.round(interface.target_pos, 3)}")
        else:
            interface.aruco_detected = False
            print("ArUco marker: Not detected (stale data)")
    else:
        print("ArUco marker: Not detected")
            


# Subscribe to camera image and command topics
rospy.Subscriber("/camera/color/image_raw", Image, image_callback)
rospy.Subscriber("/command_topic", UInt8MultiArray, command_callback)
rospy.Timer(rospy.Duration(1.0), timer_callback)

# Let ROS handle all callbacks
rospy.spin()
