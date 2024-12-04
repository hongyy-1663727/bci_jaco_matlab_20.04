import roslib
roslib.load_manifest('kinova_demo')
import rospy
import csv
import time
import datetime
import numpy as np
import sys
import struct

from robot_control_modules import *
from geometry_msgs.msg import Pose, PoseStamped, Quaternion
from kinova_msgs.msg import JointAngles
import socket

import pygame as pg
import tf.transformations  # Ensure tf is imported for transformations

class Target():
    def __init__(self, pos, q, ind): 
        self.pos = np.array(pos)
        self.q = q
        self.ind = ind

class Display():
    def __init__(self):
        pg.init()
        self.screen = pg.display.set_mode((1000, 1000))
        # pg.display.set_caption("User Display")
        # self.screen = pg.display.get_surface()
        self.screen.fill((255, 0, 0))
        pg.display.flip()
        # self.updateFeedback((100,100,100)) 
        self.font = pg.font.Font(None, 300)
        self.actionNames = ['R Thumb', 'L Leg', 'L Thumb', 'R Wrist', 'Lips', 'Tongue', 'Both']
        self.colors = [
            (233, 37, 127), 
            (244, 120, 50), 
            (254, 201, 56),
            (59, 196, 227),
            (71, 183, 73),
            (115, 52, 131),
            (175, 170, 168)
        ]

    def changeBGColor(self, col):
        # print(col)conda
        # self.screen.fill(col)
        pg.draw.rect(self.screen, col, pg.Rect(0, 500, 1000, 1000))
        pg.display.flip()

    def updateFeedback(self, col):
        pg.draw.rect(self.screen, col, pg.Rect(0, 0, 1000, 500))
        pg.display.flip()
        
    def updateText(self, dim):
        txt = self.actionNames[dim - 1]
        color = self.colors[dim - 1]

        self.screen.fill((0, 0, 0), (0, 0, 1000, 500))  
        self.txt_surface1 = self.font.render(txt, True, color)
        self.screen.blit(self.txt_surface1, (50, 200))
        pg.display.flip()

    def checkClose(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()

class DiscreteActionsRobot():
    def __init__(self, *args, **kwargs):  
        for _k, _v in kwargs.items():
            if hasattr(self, _k):
                setattr(self, _k, _v)
        rospy.init_node('testJacoInterface')

        self.userDisplay = Display()

        self.prefix = 'j2n6s300_'
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.connect(('127.0.0.1', 43210))

        self.home = [-0.20, -0.40, 0.25]
        self.matlabPos = [0, 0, 0]
        self.logOpen = False
        self.inTargetCount = 0
        self.targetDone = 0
        self.key = 0

        # Dynamics
        self.neuralDT = 1 / 5.0

        # Default translation (fast)
        self.kv_f = 0.8      # damping coefficient
        self.ki_f = 0.01800  # mass coefficient 
        
        # Slow translation
        self.kv_s = 0.7      # damping coefficient
        self.ki_s = 0.01200  # mass coefficient 
        
        # Initialize with default vals
        self.kv = self.kv_f
        self.ki = self.ki_f

        # Rotation vals
        self.gkv = 0.8  # damping coefficient
        self.gki = 0.10  # mass coefficient 

        # Initialize trans and rot vel
        self.V = np.array([0.0, 0.0, 0.0])
        self.R = np.array([0.0, 0.0, 0.0])

        # Workspace limits
        self.wl = [-0.5, -0.6, 0.1]
        self.wu = [0.05, -0.05, 0.45]
        self.latRotLim = [-1.0, 1.0]

        # Target parameters
        self.assistAlpha = 1.0
        self.k = 0.01
        self.targetDist = [100, 100, 100, 100]

        self.fv = 0.8  # damping coefficient
        self.fi = 500  # mass coefficient 

        self.fing = 0
        self.fl = 0
        self.fu = 7400
        self.FV = 0
        self.dt = 0.2

        # Running grasp parameters
        self.UseRunningGrasp = 1
        self.runningGraspBinNum = 8
        self.runningGraspThresh = 0.7
        self.runningGrasp = np.zeros(self.runningGraspBinNum)
        self.openCnt = 0
        self.closeCnt = 0
        self.openPercent = 0.0
        self.closePercent = 0.0
        self.gripperClose = 7400
        self.gripperOpen = 0
        self.euler = [0.0, 0.0, 0.0]

        self.targetBoundGripper = 500
        self.targetBoundRot = 0.3

        self.targetBoundVert = 0.03
        self.upPos = 0.4
        self.downPos = 0.2

        # Running switch parameters
        self.UseRunningSwitch = 1
        self.runningSwitchBinNum = 8
        self.runningSwitchThresh = 0.7
        self.runningSwitch = np.zeros(self.runningSwitchBinNum)
        self.switchCnt = 0
        self.switchPercent = 0.0
        self.switchInput = 7
        self.switchLocked = 0

        # Running input parameters
        self.runningInputBinNum = 5
        self.runningInputThresh = 4
        self.runningInput = np.zeros(self.runningInputBinNum)

        # Targets
        self.t0 = Target([0.1, 0.1, 0.0], -1, 0)
        self.t1 = Target([-0.2, -0.4, 0.15], 1, 1)

        self.operationalMode = 0
        self.setOperationMode()
        self.switchLockTime = 2

        self.autoCenterOverTarget = 1
        self.autoCenterDist = 0.1

        self.dist2D = 100.0

        self.gripper = 0
        self.dampenUp = 0

        self.wristStartX = 3.1415
        self.wristStartY = 0.0
        self.wristStartZ = 0.0

        self.operationalModeReset = 0
        self.lowGainMode = 0

        self.graspOrientation = 0
        self.RotThetaX = 0.0

        self.inTrial = 0

        self.goalVal = [0.0, 0.0, 0.0]

        self.modeswitch = 0
        self.UseAutoGraspTD = 0
        self.UseHeightOffset = 0
        self.AutoGraspHorzDist = 10.0
        self.AutoGraspVertDist = 10.0

        # Pose assist actions
        self.poseAssistAction = [4, 2]

        self.t1_pos0 = [100.0, 100.0, 100.0]
        self.t1_pos1 = [100.0, 100.0, 100.0]

        self.t1_gpos0 = [100.0, 100.0, 100.0]
        self.t1_gpos1 = [100.0, 100.0, 100.0]

        self.goalGrasp = 1
        self.goalPos = self.t1_pos1 
        self.AutoPoseDist = 0.15
        self.EnableGoalSwitch = 1
        self.AssistLocked = 0
        self.assistLockTime = 2

        self.disableModeSwitch = 1
        self.subPose = rospy.Subscriber('/j2n6s300_driver/out/tool_pose', PoseStamped, self.callbackPose)
        self.subJoint = rospy.Subscriber('/j2n6s300_driver/out/joint_angles', JointAngles, self.callbackJoint)
        self.reset()

    def updateRunningSwitch(self, g):
        if self.switchLocked == 1:
            t = datetime.datetime.now()
            tdelta = (t - self.lastSwitchTime).seconds
            # print("TDELTA", tdelta)
            if tdelta >= self.switchLockTime:
                self.switchLocked = 0

        if self.switchLocked == 0:
            self.runningSwitch = np.roll(self.runningSwitch, -1)
            self.runningSwitch[-1] = g

        self.switchCnt = np.count_nonzero(self.runningSwitch == self.switchInput)
        self.switchPercent = float(self.switchCnt) / float(self.runningSwitchBinNum)

    def updateRunningGrasp(self, g):
        self.runningGrasp = np.roll(self.runningGrasp, -1)
        self.runningGrasp[-1] = g

        self.openCnt = np.count_nonzero(self.runningGrasp == 1)
        self.closeCnt = np.count_nonzero(self.runningGrasp == -1)

        self.openPercent = float(self.openCnt) / float(self.runningGraspBinNum)
        self.closePercent = float(self.closeCnt) / float(self.runningGraspBinNum)

    def updateRunningInput(self, input_val):
        if self.AssistLocked == 1:
            t = datetime.datetime.now()
            tdelta = (t - self.lastAssistTime).seconds
            # print("TDELTA", tdelta)
            if tdelta >= self.assistLockTime:
                self.AssistLocked = 0

        self.runningInput = np.roll(self.runningInput, -1)
        self.runningInput[-1] = input_val

        vals, counts = np.unique(self.runningInput, return_counts=True)
        if np.max(counts) > self.runningInputThresh: 
            self.longInput = vals[np.argmax(counts)]
        else:
            self.longInput = 0

    def reset(self):
        position = self.home
        # print(position)
        self.operationalMode = self.operationalModeReset
        self.setOperationMode()
        self.userDisplay.screen.fill((0, 0, 0), (0, 0, 1000, 500))
        pg.display.flip()
        quaternion = tf.transformations.quaternion_from_euler(
            self.wristStartX, self.wristStartY, self.wristStartZ, 'rxyz'
        )  
        result = cartesian_pose_client(position, quaternion, self.prefix)
        self.runningGrasp = np.zeros(self.runningGraspBinNum)
        time.sleep(1)
        self.setGripper(0)
        self.gripper = 0
        time.sleep(0.1)

        self.R = np.array([0.0, 0.0, 0.0])
        self.V = np.array([0.0, 0.0, 0.0])

        self.kv = self.kv_f
        self.ki = self.ki_f

        self.dist2D = 100.0
        self.agStep = 0
        self.t1 = 1
        self.t2 = 1
        self.initialApproach = 1
        self.graspInit = 0
        self.graspGo = 0
        self.graspGoal = 0
        self.EnableGoalSwitch = 1
        if self.logOpen:
            self.file.close()
            self.logOpen = False
            self.inTrial = 0

    def stopRobot(self):
        self.setVelocity([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        
    def setVelocity(self, V, R):
        duration_sec = 0.18
        # print(error)
        p = 1.0
        self.updateLogger()
        # self.checkInTargetGrasp()
        # print(R)
        publishCatesianVelocityCommands(
            [V[0], V[1], V[2], R[0], R[1], R[2]], duration_sec, self.prefix
        )

    def callbackPose(self, msg):
        self.pose = [
            msg.pose.position.x, 
            msg.pose.position.y, 
            msg.pose.position.z
        ]

        self.quaternion = (
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        )
        self.euler = tf.transformations.euler_from_quaternion(self.quaternion)
        # print('EUL:', self.euler)
        self.t0.pos = self.pose
        # print(self.euler)

    def callbackJoint(self, msg):
        self.jointAngles = (
            msg.joint1,
            msg.joint2,
            msg.joint3,
            msg.joint4,
            msg.joint5,
            msg.joint6,
        )
        # print(self.jointAngles[4] - self.jointAngles[3])

    def setMode(self, mode):
        self.mode = mode
        # self.home = [-0.2, -0.4, self.matlabPos[2]] 
        # if self.mode == 5 or self.mode > 7:
        self.home = [
            self.matlabPos[0] - 0.2,
            self.matlabPos[1] - 0.4,
            self.matlabPos[2]
        ]
        print("home: ", self.home)
        self.reset()

    def startLogger(self, fn):
        self.file = open(fn, "w", newline='')  # Added newline='' for CSV
        self.fileObj = csv.writer(self.file)
        self.logOpen = True
        self.logT = time.time()

    def updateLogger(self):
        line = [
            time.time(), 
            self.pose[0], 
            self.pose[1], 
            self.pose[2], 
            self.V[0], 
            self.V[1], 
            self.V[2], 
            self.fing, 
            self.euler[0],
            self.euler[1], 
            self.euler[2],
            self.operationalMode, 
            self.key, 
            self.R[0], 
            self.R[1], 
            self.R[2]
        ]
        self.fileObj.writerow(line)

    def checkInTargetGrasp(self):
        if (self.TargetID == 1) and (abs(self.fing - self.gripperClose) < self.targetBoundGripper):
            self.inTarget = 1
            self.inTargetCount += 1
        elif (self.TargetID == 3) and (abs(self.fing - self.gripperOpen) < self.targetBoundGripper):
            self.inTarget = 1   
            self.inTargetCount += 1
        elif (self.TargetID == 2) and (abs(abs(self.euler[2]) - 3.14) < self.targetBoundRot):
            print("IN TARGET")
            self.inTarget = 1   
            self.inTargetCount += 1 
        elif (self.TargetID == 4) and (abs(self.euler[2]) < self.targetBoundRot):
            print("IN TARGET")
            self.inTarget = 1   
            self.inTargetCount += 1 
        elif (self.TargetID == 5) and (abs(self.pose[2] - self.upPos) < self.targetBoundVert):
            print("IN TARGET")
            self.inTarget = 1   
            self.inTargetCount += 1 
        elif (self.TargetID == 6) and (abs(self.pose[2] - self.downPos) < self.targetBoundVert):
            print("IN TARGET")
            self.inTarget = 1   
            self.inTargetCount += 1 
        else:
            self.inTarget = 0   
            self.inTargetCount = 0
        
        if (self.targetDone == 0) and (self.inTargetCount > self.holdTimeSteps):
            print("Done")

            message = struct.pack("B", 2)
            self.sock.send(message)
            self.targetDone = 1

    def goToPosVel(self, position):
        vel = 0.05
        duration_sec = 0.05
        vTarget = np.array(position) - np.array(self.pose)
        vTargetNorm = np.linalg.norm(vTarget)

        while vTargetNorm > 0.01:
            # print(vTargetNorm)
            norm_vTarget = vTarget / vTargetNorm
            V = norm_vTarget * vel
            publishCatesianVelocityCommands(
                [V[0], V[1], V[2], 0.0, 0.0, 0.0], duration_sec, self.prefix
            )
            vTarget = np.array(position) - np.array(self.pose)
            vTargetNorm = np.linalg.norm(vTarget)
        V = [0.0, 0.0, 0.0]
        publishCatesianVelocityCommands(
            [V[0], V[1], V[2], 0.0, 0.0, 0.0], duration_sec, self.prefix
        )

    def setGripper(self, f):
        # Workaround for getting stuck
        if (
            abs(self.jointAngles[4] - self.jointAngles[3]) < 5 or
            abs(abs(self.jointAngles[4] - self.jointAngles[3]) - 360) < 5 or
            abs(abs(self.jointAngles[4] - self.jointAngles[3]) - 180) < 5
        ):
            m = (self.jointAngles[4] + self.jointAngles[3]) * 0.5
            s = np.sign(self.jointAngles[4] - self.jointAngles[3]) 
            goalAng = (
                self.jointAngles[0],
                self.jointAngles[1],
                self.jointAngles[2],
                self.jointAngles[3] - s * 5,
                self.jointAngles[4] + s * 5,
                self.jointAngles[5],
                0.0
            )       
            result = joint_position_client(goalAng, self.prefix)
            time.sleep(0.1)
            print("NUDGE")
        
        self.fing = max(f, self.fl)
        self.fing = min(f, self.fu)
        self.fing = round(self.fing)

        self.fingers = [self.fing, self.fing, self.fing]
        result = gripper_client(self.fingers, self.prefix)
        time.sleep(0.1)

        self.goalMet = 1
        # print(self.fing)
        # result = cartesian_pose_client(self.pose, self.quaternion, self.prefix)

    def inputToAction(self, input_val):
        # print(input_val)
        # print('EUL: ', self.euler)

        # print('target ', self.target_pos)
        # print('pos ', self.pose)
        self.modeswitch = 0
        if self.UseRunningSwitch:
            self.updateRunningSwitch(input_val)
            if self.switchPercent > self.runningSwitchThresh:
                self.switchModes()

        if self.UseAutoGraspTD == 1:
            # Determine target
            d1 = self.distance(self.t1_pos, self.pose)
            d2 = self.distance(self.t2_pos, self.pose)

            print("dist ", d1, d2)
            # print("possible target: ", self.t1, self.t2)
            if self.t1 and self.t2:
                if d1 < d2:
                    self.target_pos = self.t1_pos
                    self.activeTarget = 1
                else:
                    self.target_pos = self.t2_pos
                    self.activeTarget = 2
            elif self.t1:
                self.target_pos = self.t1_pos
                self.activeTarget = 1
            elif self.t2: 
                self.target_pos = self.t2_pos
                self.activeTarget = 2
            else:
                self.UseAutoGraspTD = 0

            self.distance2D(self.target_pos, self.pose)
            # print('2d', self.dist2D, " Z ", -self.distZ, "gripper: ", self.gripper, " mode: ", self.operationalMode)
            if (
                self.dist2D < self.AutoGraspHorzDist and 
                self.distZ > -self.AutoGraspVertDist and 
                self.operationalMode < 3 and 
                self.gripper == 0
            ):
                self.operationalMode = 3
                self.setOperationMode()
                self.switchLocked = 0

        elif self.UseAutoGraspTD == 2:  # Pose assist
            d1 = self.distance(self.t1_pos0, self.pose)
            d2 = self.distance(self.t1_pos1, self.pose)

            dist = min(d1, d2)

            if self.initialApproach == 1:
                if (dist < self.AutoPoseDist) and (self.gripper == 0):
                    self.runningInput = np.zeros(self.runningInputBinNum)

                    self.initialApproach = 0
                    self.operationalMode = 4
                    self.setOperationMode()
                    self.switchLocked = 0
                    self.AssistLocked = 1
                    self.lastAssistTime = datetime.datetime.now()
                    print(self.pose[2])
                    if d1 < d2:
                        print("SIDE")
                        self.goalGrasp = 0
                        # self.goalEul = 3.14/2
                        self.goalEul = 3.141
                        self.goalPos = self.t1_pos0
                    else:
                        print("TOP")
                        self.goalGrasp = 1
                        self.goalEul = 3.14
                        self.goalPos = self.t1_pos1

        # print("MODE ", self.operationalMode)
        if self.disableModeSwitch:
            self.inputToXYZVel(input_val)
        else:
            if self.operationalMode in [0, 2]:
                self.inputToXYZVel(input_val)
            elif self.operationalMode == 1:
                self.inputToWristGrasp(input_val)
            elif self.operationalMode == 3:  # top-down autograsp
                self.autoGraspTopDown(input_val)    
            elif self.operationalMode == 4:  # post assist top-lat
                self.updateRunningInput(input_val)
                # self.inputToPoseAssistEul(input_val)
                self.inputToPoseAssistPos(input_val)

    def poseAssistGrasp(self):
        self.distance(self.goalPos, self.pose)
        vTarget = self.dist3Dvec
        vTargetNorm = self.dist3D

        # print("Dist: ", vTargetNorm)
        if vTargetNorm > 0.01:
            norm_vTarget = vTarget
            self.gposGoalMet = 0
        else:
            # print("ABOVE TARGET")
            vTarget = np.array([0.0, 0.0, 0.0])
            self.gposGoalMet = 1

        self.V[0] = self.assistAlpha * vTarget[0]
        self.V[1] = self.assistAlpha * vTarget[1]
        self.V[2] = self.assistAlpha * vTarget[2]
        print("mode", self.gposGoalMet)
        if self.gposGoalMet == 1:
            self.setGripper(self.gripperClose)
            self.gripper = 1
            self.operationalMode = 0
            self.setOperationMode()

    def inputToPoseAssistEul(self, input_val):
        if self.EnableGoalSwitch == 1:
            if self.longInput == self.poseAssistAction[0]:
                self.goalGrasp = 0
                self.goalEul = 3.1
                self.graspOrientation = 0

            if self.longInput == self.poseAssistAction[1]:
                self.goalGrasp = 1
                self.goalEul = 3.14
                self.graspOrientation = 0

        self.goalEul = 3.14
        self.graspOrientation = 0

        if self.goalGrasp == 0:
            if abs(self.euler[0]) > (3.14 / 2):
                self.R[0] = -0.4
            else:
                self.R[0] = 0

        elif self.goalGrasp == 1:
            print("ang dist ", abs(abs(self.euler[0]) - 3.14))
            if abs(abs(self.euler[0]) - 3.14) > 0.1:
                self.R[0] = np.sign(self.euler[0]) * 0.4
            else:
                self.R[0] = 0
                self.PoseAssistGoalMet = 1

    def inputToPoseAssistPos(self, input_val):
        if self.EnableGoalSwitch == 1:
            if self.longInput == self.poseAssistAction[0]:
                self.goalGrasp = 0
                self.goalPos = self.t1_pos0
                
            if self.longInput == self.poseAssistAction[1]:
                self.goalGrasp = 1
                self.goalPos = self.t1_pos1

        self.distance(self.goalPos, self.pose)
        vTarget = self.dist3Dvec
        vTargetNorm = self.dist3D

        # print("Dist: ", vTargetNorm)
        if vTargetNorm > 0.01:
            norm_vTarget = vTarget
            self.posGoalMet = 0
        else:
            # print("ABOVE TARGET")
            vTarget = np.array([0.0, 0.0, 0.0])
            self.posGoalMet = 1

        self.V[0] = self.assistAlpha * vTarget[0]
        self.V[1] = self.assistAlpha * vTarget[1]
        self.V[2] = self.assistAlpha * vTarget[2]

        print("dist: ", vTargetNorm)
        print("goalGrasp ", self.goalGrasp)

    def autoGraspTopDown(self, inp):
        # Autocenter over target
        if self.agStep == 0:
            trans_done = 0
            ang_done = 0
            self.distance2D(self.target_pos, self.pose)
            vTarget = self.dist2Dvec
            vTargetNorm = self.dist2D

            # print("Dist: ", vTargetNorm)
            if vTargetNorm > 0.01:
                norm_vTarget = vTarget
            else:
                # print("ABOVE TARGET")
                vTarget = np.array([0.0, 0.0, 0.0])
                trans_done = 1

            ang = self.euler[2]
            print("AT: ", self.activeTarget)
            if self.activeTarget == 1:
                goal_ang = -3.14 / 2
            else:
                goal_ang = -3.14 / 4

            ang_dist = ang - goal_ang

            print("ANG DIST: ", ang_dist)

            if abs(ang_dist) < 0.1 or (abs(abs(ang_dist) - 3.14) < 0.1):
                ang_done = 1
                self.R[2] = 0
                print("DONE")  
            else:
                if ang_dist < 0:
                    self.R[2] = -0.5
                else:
                    self.R[2] = 0.5
                print("ROTATE")     

            self.V[0] = 0.75 * self.assistAlpha * vTarget[0]
            self.V[1] = 0.75 * self.assistAlpha * vTarget[1]
            self.V[2] = 0.0

            if trans_done and ang_done:
                self.agStep = 2

        elif self.agStep == 2:
            print("continue?")
            if self.WaitForGraspSignal:
                if inp == 1:
                    self.agStep = 3
                else:
                    self.agStep = 2
            elif self.WaitForGraspSignal == 0:
                self.agStep = 3

        elif self.agStep == 3:
            self.distance2D(self.target_pos, self.pose)
            # print(self.distZ)
            if self.distZ < 0:
                self.V[2] = -self.assistAlpha * 1.5
            else:
                self.V[2] = 0
                self.agStep = 4
        elif self.agStep == 4:
            self.agStep = 0
            self.setGripper(self.gripperClose)
            self.gripper = 1
            self.operationalMode = 0
            self.setOperationMode()
            
            if self.activeTarget == 1:
                self.t1 = 0
            else:
                self.t2 = 0

    def inputToWristGrasp(self, input_val):
        self.key = input_val

        u = np.array([0.0, 0.0, 0.0])
        u[1] = -(int(input_val == 1) - int(input_val == 3))
        u[0] = -(int(input_val == 2) - int(input_val == 4))
        u[2] = int(input_val == 5) - int(input_val == 6)

        u[0] = 0  # Disable rotation for test

        if self.graspOrientation == 0:  # top-down
            if self.dampenUp and self.gripper == 0:
                u[2] = 0.2 * int(input_val == 5) - int(input_val == 6)

            self.V[0] = 0.0
            self.V[1] = 0.0
            self.V[2] = self.kv * self.V[2] + self.ki * (u[2])
            self.R[2] = self.gkv * self.R[2] + -self.gki * (u[0])
        elif self.graspOrientation == 1:
            theta = self.euler[2]

            vy = np.cos(theta) * u[2]
            vx = -np.sin(theta) * u[2]

            self.V[0] = self.kv * self.V[0] + self.ki * (vx)
            self.V[1] = self.kv * self.V[1] + self.ki * (vy)
            self.V[2] = 0.0
            self.R[1] = self.gkv * self.R[1] + -self.gki * (u[0])

            if self.euler[2] <= self.latRotLim[0]:
                print("ROTATION LIMIT 1")
                self.R[1] = max(self.R[1], 0.0)
            elif self.euler[2] >= self.latRotLim[1]:
                print("ROTATION LIMIT 2")
                self.R[1] = min(self.R[1], 0.0)

        # Workspace limits
        if self.pose[0] >= self.wu[0]:
            self.V[0] = min(self.V[0], 0.0)
        
        if self.UseRunningGrasp:
            g = u[1]
            self.updateRunningGrasp(g)
        
            if self.closePercent > self.runningGraspThresh:
                self.setGripper(self.gripperClose)
                self.gripper = 1
                self.runningGrasp = np.zeros(self.runningGraspBinNum)
            elif self.openPercent > self.runningGraspThresh:
                self.setGripper(self.gripperOpen)
                self.runningGrasp = np.zeros(self.runningGraspBinNum)
                self.gripper = 0
        else:
            self.FV = self.fv * self.FV + self.fi * u[1]
            self.fing = self.fing + self.FV * self.dt
            self.setGripper(self.fing)

        # Workspace limits
        if self.pose[0] <= self.wl[0]:
            self.V[0] = max(self.V[0], 0.0)
        if self.pose[0] >= self.wu[0]:
            self.V[0] = min(self.V[0], 0.0)
        if self.pose[1] <= self.wl[1]:
            self.V[1] = max(self.V[1], 0.0)
        if self.pose[1] >= self.wu[1]:
            self.V[1] = min(self.V[1], 0.0)
        if self.pose[2] <= self.wl[2]:
            self.V[2] = max(self.V[2], 0.0)
        if self.pose[2] >= self.wu[2]:
            self.V[2] = min(self.V[2], 0.0)

    def setOperationMode(self):
        if self.operationalMode == 1:
            self.runningSwitch = np.zeros(self.runningSwitchBinNum)
            self.userDisplay.changeBGColor((0, 0, 255))
            print("Mode: Grasp")
            self.kv_s = 0.7    # damping coefficient
            self.ki_s = 0.01200  # mass coefficient 

            if self.autoCenterOverTarget == 1 and (self.dist2D < self.autoCenterDist):
                print("CENTER OVER TARGET")
                position = [self.t1.pos[0], self.t1.pos[1], self.pose[2]]
                # print(position)
                result = cartesian_pose_client(position, self.quaternion, self.prefix)
        elif self.operationalMode == 0:
            self.runningSwitch = np.zeros(self.runningSwitchBinNum)
            self.userDisplay.changeBGColor((255, 0, 0))
            self.kv = self.kv_f
            self.ki = self.ki_f
            print("Mode: Translation")

        elif self.operationalMode == 2:
            self.runningSwitch = np.zeros(self.runningSwitchBinNum)
            self.userDisplay.changeBGColor((0, 255, 0))
            self.kv = self.kv_s
            self.ki = self.ki_s
            print("Mode: Low Gain Translation")

        elif self.operationalMode == 3:
            self.runningSwitch = np.zeros(self.runningSwitchBinNum)
            self.userDisplay.changeBGColor((255, 255, 0))
            self.kv = self.kv_s
            self.ki = self.ki_s
            print("Mode: Auto Grasp")

        elif self.operationalMode == 4:
            self.runningSwitch = np.zeros(self.runningSwitchBinNum)
            self.userDisplay.changeBGColor((255, 0, 255))
            self.kv = self.kv_s
            self.ki = self.ki_s
            print("Mode: Auto Pose")

    def checkPathGoal(self):
        if self.operationalMode == 0:
            if self.goalDim == 1:
                if self.pose[1] <= self.goalVal[1]:
                    self.goalMet = 1
            elif self.goalDim == 3:
                if self.pose[1] >= self.goalVal[1]:
                    self.goalMet = 1
            elif self.goalDim == 2:
                if self.pose[0] <= self.goalVal[0]:
                    self.goalMet = 1
            elif self.goalDim == 4:
                if self.pose[0] >= self.goalVal[0]:
                    self.goalMet = 1
            if self.goalDim == 5:
                if self.pose[2] >= self.goalVal[2]:
                    self.goalMet = 1
            if self.goalDim == 6:
                if self.pose[2] <= self.goalVal[2]:
                    self.goalMet = 1

        if self.operationalMode == 1:
            if self.goalDim == 2:
                if self.euler[2] >= self.goalVal[0]:
                    self.goalMet = 1
            if self.goalDim == 4:
                if self.euler[2] <= self.goalVal[0]:
                    self.goalMet = 1
            elif self.goalDim == 5:
                if self.pose[1] >= self.goalVal[1]:
                    self.goalMet = 1
            elif self.goalDim == 6:
                if self.pose[1] <= self.goalVal[1]:
                    self.goalMet = 1

    def switchModes(self):
        print("SWITCH MODES", self.operationalMode)
        if self.operationalMode == 0: 
            if self.lowGainMode == 1:
                self.operationalMode = 2  # slow
            else:
                self.operationalMode = 1  # grasp
        elif self.operationalMode == 1:
            self.operationalMode = 0
        elif self.operationalMode == 2:    # slow
            self.operationalMode = 1
        elif self.operationalMode == 3:    # assist
            self.operationalMode = 0
            if self.activeTarget == 1:
                self.t1 = 0
            else:
                self.t2 = 0
        elif self.operationalMode == 4:    # assist
            self.operationalMode = 1

        self.goalMet = 1   
        self.setOperationMode()
        self.lastSwitchTime = datetime.datetime.now()
        self.switchLocked = 1
        self.modeswitch = self.operationalMode + 1

    def inputToXYZVel(self, input_val):
        self.key = input_val
        u = np.array([0.0, 0.0, 0.0, 0.0])

        if self.view == 1:
            u[1] = -(int(input_val == 1) - int(input_val == 3))
            u[0] = -(int(input_val == 2) - int(input_val == 4))
            u[2] = int(input_val == 5) - int(input_val == 6)
            u[3] = int(input_val == 8) - int(input_val == 9)
        elif self.view == 2:
            u[1] = -(int(input_val == 3) - int(input_val == 1))
            u[0] = -(int(input_val == 4) - int(input_val == 2))
            u[2] = int(input_val == 5) - int(input_val == 6)
            u[3] = int(input_val == 8) - int(input_val == 9)
        elif self.view == 3:
            u[1] = -(int(input_val == 4) - int(input_val == 2))
            u[0] = -(int(input_val == 1) - int(input_val == 3))
            u[2] = int(input_val == 5) - int(input_val == 6)
            u[3] = int(input_val == 8) - int(input_val == 9)

        self.distance2D(self.target_pos, self.pose)
        vTarget = self.dist2Dvec
        vTargetNorm = self.dist2D

        if vTargetNorm > 0.005:
            norm_vTarget = vTarget
        else:
            norm_vTarget = np.array([0.0, 0.0, 0.0])

        alpha = self.assistAlpha
        AssistVel = self.ki * alpha * norm_vTarget

        self.V[0] = self.kv * self.V[0] + (1 - alpha) * self.ki * u[0] + AssistVel[0]
        self.V[1] = self.kv * self.V[1] + (1 - alpha) * self.ki * u[1] + AssistVel[1]
        self.V[2] = self.kv * self.V[2] + self.ki * u[2]

        # Rotation X
        self.R[0] = self.gkv * self.R[0] + -self.gki * (u[3])
        self.R[2] = 0.0

        # Workspace limits
        if self.pose[0] <= self.wl[0]:
            self.V[0] = max(self.V[0], 0.0)
        if self.pose[0] >= self.wu[0]:
            self.V[0] = min(self.V[0], 0.0)
        if self.pose[1] <= self.wl[1]:
            self.V[1] = max(self.V[1], 0.0)
        if self.pose[1] >= self.wu[1]:
            self.V[1] = min(self.V[1], 0.0)
        if self.pose[2] <= self.wl[2]:
            self.V[2] = max(self.V[2], 0.0)
        if self.pose[2] >= self.wu[2]:
            self.V[2] = min(self.V[2], 0.0)

    def flipStop(self):
        print("flip STOPPING")
        self.stopRobot()
        self.operationalMode = 2
        self.setOperationMode()

    def distance(self, p1, p2):
        a = np.array(p1[:3])
        b = np.array(p2[:3])
        d = np.linalg.norm(a - b)       
        if d != 0:
            dhat = (a - b) / d
        else:
            dhat = np.array([0.0, 0.0, 0.0])
        self.dist3D = d
        self.dist3Dvec = dhat
        return d

    def distance2D(self, p1, p2):
        a = np.array([p1[0], p1[1]])
        b = np.array([p2[0], p2[1]])
        d = np.linalg.norm(a - b)
        self.dist2D = d
        if d != 0:
            self.dist2Dvec = (a - b) / d
        else:
            self.dist2Dvec = np.array([0.0, 0.0])
        self.distZ = p1[2] - p2[2]





