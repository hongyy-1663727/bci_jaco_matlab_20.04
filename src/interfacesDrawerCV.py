import roslib; roslib.load_manifest('kinova_demo')
import rospy
import csv
import time
import datetime
import numpy as np
import tf

from cv_bridge import CvBridge

from robot_control_modules import *
from geometry_msgs.msg import Pose, PoseStamped, Quaternion
from kinova_msgs.msg import JointAngles
import socket
import torch
import cv2
from ultralytics import YOLOWorld
from sensor_msgs.msg import Image
import tf2_ros
import tf2_geometry_msgs

import pygame as pg
import pandas as pd



class Target():
    def __init__(self, pos, q, ind): 
        self.pos = np.array(pos)
        self.q   = q
        self.ind = ind

class GoalObject():
    """ a class for goal objects

    Attributes
    ---------
        pos: position as read by camera
        ppos: position of preview (offset away from grasp pos)
        gpos: grasp position
        opos: open position (offset away from grasp pos)
        q: quaternion representation of grasp orientation
        openRoutine: boolean 0: drawer, 1: door
        SP: list of subposes in waypoints
        nn: nearest neighbor goals
    """
    def __init__(self, pos, ratio, lr): 
        """
        Parameters
        ---------
        pos : position 
        ratio: aspect ratio of bounding box
        lr: left-right position of handle within surroundign cabinet
        """
        pos[2] = pos[2] + .02

        self.pos    = np.array(pos)
        self.pos[1] = self.pos[1] + 0.05

        # for side approach
        self.ppos   = np.array(self.pos)
        self.ppos[1]   = self.ppos[1] + 0.1

        self.gpos   = np.array(self.pos)
        self.gpos[1]   = self.gpos[1] - 0.04

        self.opos   = np.array(pos)
        self.opos[1]   = self.opos[1] + 0.21
        
        # set orientation based on aspect ratio of bounding box
        if ratio > 2:
            self.q = [0.508758544921875, 0.4937331974506378, -0.4914683699607849, 0.5058171153068542]
        else:
            self.q = [0.708938717842102, -0.003660775488242507, 0.0032196040265262127, 0.7052531838417053]

        self.nn = np.array([-1,-1,-1,-1,-1,-1,-1]) # nearest neighbor in each direction

        open_pos = self.opos

        # set subroutine based on left-right ratio of handle
        if lr > 0.35 and lr < 0.65: # drawer
            self.openRoutine = 0

            sub_pos1 = [open_pos[0],  open_pos[1], open_pos[2]]
            sub_pos1[1] = sub_pos1[1] + 0


            sub_pos1a = sub_pos1[:]
            sub_pos1a[1] = sub_pos1a[1] + 0.01

            sub_pos2 = sub_pos1a[:]
            sub_pos2[2] = sub_pos2[2] + 0.18

            sub_pos3 = sub_pos2[:]
            sub_pos3[1] = sub_pos3[1] 

            sub_pos4 = sub_pos3[:]
            sub_pos4[1] = sub_pos4[1]- 0.10

            sub_pos5 = sub_pos4[:]
            sub_pos4[1] = sub_pos4[1]- 0.10
            sub_pos5[2] = sub_pos5[2]- 0.08

            self.SP = [sub_pos1,sub_pos1a, sub_pos2, sub_pos3, sub_pos4, sub_pos5]
            self.QP =  [0.9999791979789734, -0.0021352891344577074, -0.0025787334889173508, 0.0055120266042649752]

        else: # door
            self.openRoutine = 1 

            self.opos   = np.array(pos)
            self.opos[1]   = self.opos[1] + 0.12
            open_pos = self.opos

            sub_pos1 = [open_pos[0],  open_pos[1], open_pos[2]]
            sub_pos1[1] = sub_pos1[1] + 0.01
            sub_pos1[0] = sub_pos1[0] - 0.03


            sub_pos1a = sub_pos1[:]
            sub_pos1a[1] = sub_pos1a[1] + 0.01

            sub_pos2 = sub_pos1a[:]
            sub_pos2[0] = sub_pos2[0] + 0.18

            sub_pos3 = sub_pos2[:]
            sub_pos3[1] = sub_pos3[1] - 0.14

            sub_pos4 = sub_pos3[:]
            sub_pos4[0] = sub_pos4[0]- 0.1

            sub_pos4a = sub_pos4[:]
            sub_pos4a[1] = sub_pos4a[1] + 0.1

            sub_pos5 = sub_pos4a[:]
            sub_pos5[0] = sub_pos5[0]- 0.05
            sub_pos5[1] = sub_pos5[1]+ 0.05

            sub_pos5a = sub_pos5[:]
            sub_pos5a[1] = sub_pos5a[1] - 0.05

            sub_pos6 = sub_pos5[:]
            sub_pos6[0] = sub_pos6[0]- 0.20

            sub_pos7 = sub_pos6[:]
            sub_pos7[0] = sub_pos7[0] + 0.10
            sub_pos7[1] = sub_pos7[1] - 0.12
            sub_pos7[2] = sub_pos7[2] - 0.02

            self.SP = [sub_pos1, sub_pos1a, sub_pos2, sub_pos3, sub_pos4, sub_pos4a, sub_pos5, sub_pos5a, sub_pos6, sub_pos7]

    def setNeighbor(self, ind, val):
        self.nn[ind] = val

class Display():
    """class for user display"""
    def __init__(self):
        pg.init()
        self.screen = pg.display.set_mode((1000,1000))
        self.screen.fill((255,0,0))
        pg.display.flip()
        self.font = pg.font.Font(None, 300)
        self.actionNames = ['R Thumb', 'L Leg', 'L Thumb', 'Head', 'Lips', 'Tongue', 'Middle' ]
        self.colors = [(233, 37, 127), (244,120,50), (254, 201, 56),(59, 196, 227),(71, 183, 73),(115, 52, 131),(175, 170, 168)]

    def changeBGColor(self, col):
        # print(col)conda
        # self.screen.fill(col)
        pg.draw.rect(self.screen, col, pg.Rect(0,500,1000,1000))
        pg.display.flip()

    def updateFeedback(self, col):
        pg.draw.rect(self.screen, col, pg.Rect(0,0,1000,500))
        pg.display.flip()
        
    def updateText(self, dim):
        txt = self.actionNames[dim-1]
        color = self.colors[dim-1]
        self.screen.fill((0,0,0), (0,0,1000, 500))  
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
        
        # --- Vision (YOLO handle detection) ---
        self.bridge = CvBridge()
        self.rgb_image = None
        self.depth_image = None
        self.raw_depth = None
        self.latest_handle_center = None

        if torch.cuda.is_available():
            self.device = 0
            rospy.loginfo(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = 'cpu'
            rospy.logwarn("GPU not available, using CPU")

        self.model = YOLOWorld("yolov8x-worldv2.pt")
        self.model.to(self.device)
        self.text_prompts = ["black handle", "cup"]
        self.model.set_classes(self.text_prompts)
        rospy.loginfo("YOLO World model loaded")

        rospy.Subscriber("/camera/color/image_raw", Image, self.rgb_cb, queue_size=1, buff_size=2**24)
        rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_cb, queue_size=1, buff_size=2**24)

        self.conf_threshold = 0.2
        self.iou_threshold = 0.4
        self.inference_times = []
        self.frame_count = 0

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.last_rgb = rospy.Time.now()


        self.prefix     = 'j2n6s300_'
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.connect(('127.0.0.1', 43210))

        self.home       = [-0.20,-0.40, 0.35]
        self.matlabPos  = [0,0,0]
        self.logOpen    = False
        self.inTargetCount = 0
        self.targetDone = 0
        self.key = 0

        # Dynamics
        self.neuralDT = 1/5

        # Default translation (fast)
        self.kv_f = 0.8      # damping coefficient
        self.ki_f = .01800   # mass coefficient 
        
        # Slow translation
        self.kv_s = 0.7    # damping coefficient
        self.ki_s = .01200   # mass coefficient 
        
        # Initialize with default vals
        self.kv = self.kv_f
        self.ki = self.ki_f

        self.gkv = 0.8; # damping coefficient
        self.gki = .10 # mass coefficient 

        # Initialize trans and rot vel
        self.V = np.array([0., 0., 0.])
        self.uV = np.array([0., 0., 0.])
        self.R = np.array([0., 0., 0.])

        # Workspace limits
        self.wl = [-0.5, -0.6, .1]
        self.wu = [0.05, -0.05, 0.45]
        self.latRotLim = [-1.0, 1.0]

        # target
        self.assistAlpha = 1.0
        self.k = .01
        self.targetDist = [100,100,100,100]

        self.fv = 0.8; # damping coefficient
        self.fi = 500# mass coefficient 

        self.fing = 0
        self.fl = 0
        self.fu = 7400
        self.handlegripper = 4000
        self.FV = 0
        self.dt = 0.2

        self.UseRunningGrasp = 1
        self.runningGraspBinNum = 8
        self.runningGraspThresh = 0.7
        self.runningGrasp = np.zeros(self.runningGraspBinNum)
        self.openCnt = 0
        self.closeCnt = 0
        self.openPercent = 0
        self.closePercent = 0
        self.gripperClose = 7400
        self.gripperOpen = 1000
        self.euler = [0,0,0]

        self.targetBoundGripper = 500
        self.targetBoundRot = 0.3

        self.targetBoundVert = .03
        self.upPos = 0.4
        self.downPos = 0.2

        self.UseRunningSwitch = 1
        self.runningSwitchBinNum = 8
        self.runningSwitchThresh = 0.7
        self.runningSwitch = np.zeros(self.runningSwitchBinNum)
        self.switchCnt = 0
        self.switchPercent = 0
        self.switchInput = 7
        self.switchLocked = 0

        self.runningInputBinNum = 8
        self.runningInputThresh = 6
        self.runningInput = np.zeros(self.runningInputBinNum)

        self.t0 = Target([.1, .1, 0], -1,0)
        self.t1 = Target([-.2, -.4, .15], 1,1)

        self.operationalMode = 0
        # self.setOperationMode()
        self.switchLockTime = 2

        self.autoCenterOverTarget = 1
        self.autoCenterDist = .1

        self.dist2D = 100

        self.gripper = 0
        self.dampenUp = 0

        self.wristStartX = 3.1415/2
        self.wristStartY = 0.0
        # self.wristStartZ = 0.5*3.1415
        self.wristStartZ = 0

        self.operationalModeReset = 0
        self.lowGainMode = 0

        self.graspOrientation = 0
        self.RotThetaX = 0

        self.inTrial = 0

        self.goalVal = [0.0, 0.0, 0]

        self.modeswitch = 0
        self.UseAutoGraspTD = 0
        self.UseHeightOffset = 0
        self.AutoGraspHorzDist     = 10
        self.AutoGraspVertDist     = 10

        self.AssistFn = 1
        self.poseAssistAction = [4, 2]

        self.t1_pos0 = [100, 100, 100]
        self.t1_pos1 = [100, 100, 100]

        self.t1_gpos0 = [100, 100, 100]
        self.t1_gpos1 = [100,100,100]
        # self.poseAssistAction = [100, 100]

        self.goalGrasp              = 1
        self.goalPos                = self.t1_pos1 
        self.AutoPoseDist           = 0.15
        self.EnableGoalSwitch       = 1
        self.AssistLocked           = 0
        self.assistLockTime         = 2

        self.AssistMode = 2
        self.UseModeSwitch = 1

        self.TaskMode = 1  # 1: R2G, 2: T2P

        self.beliefThresh = 0.4
        self.distB = 0.4
        self.distK = 1.0
        self.velB = 0.4
        self.velK = 1.0
        self.pDiag = 0.8

        self.logOpen2 = 0
        subPose     = rospy.Subscriber('/j2n6s300_driver/out/tool_pose', PoseStamped, self.callbackPose)
        subJoint    = rospy.Subscriber('/j2n6s300_driver/out/joint_angles', JointAngles, self.callbackJoint)
        self.reset() 
        
    def rgb_cb(self, msg):
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.last_rgb = rospy.Time.now()

    def depth_cb(self, msg):
        raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self.raw_depth = raw.copy()
        depth_m = raw.astype(np.float32) / 1000.0
        depth_m[np.isnan(depth_m)] = 0.0

        valid = (depth_m >= 0.1) & (depth_m <= 3.0)
        norm = np.zeros_like(depth_m, dtype=np.uint8)
        norm[valid] = np.clip(((depth_m[valid] - 0.1) / (3.0 - 0.1)) * 255, 0, 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        depth_colored[~valid] = (0, 0, 0)
        self.depth_image = depth_colored
        
    def detect_color_buttons(self, image_bgr):
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        # --- RED BUTTON MASK ---
        lower_red = np.array([0, 50, 150])
        upper_red = np.array([10, 255, 255])   
        red_mask = cv2.inRange(hsv, lower_red, upper_red)

        # --- BLUE BUTTON MASK ---
        lower_blue = np.array([90, 50, 150])
        upper_blue = np.array([115, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Clean noise
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
        
        # Optional debug: visualize mask pixels
        debug_mask = cv2.bitwise_or(red_mask, blue_mask)
        cv2.imshow("Color Debug Mask", debug_mask)
        
        red_centers = []
        blue_centers = []

        red_cnts, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in red_cnts:
            if cv2.contourArea(cnt) > 300:
                x, y, w, h = cv2.boundingRect(cnt)
                cx, cy = x + w // 2, y + h // 2
                red_centers.append((cx, cy))
                cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(image_bgr, "Red Button", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        blue_cnts, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in blue_cnts:
            if cv2.contourArea(cnt) > 300:
                x, y, w, h = cv2.boundingRect(cnt)
                cx, cy = x + w // 2, y + h // 2
                blue_centers.append((cx, cy))
                cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(image_bgr, "Blue Button", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return image_bgr, red_centers, blue_centers
            
    def vision_inference_step(self):
        if self.rgb_image is None:
            # Create a waiting message window if no image is available
            waiting_img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(waiting_img, "Waiting for camera data...", (50, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Camera Feed", waiting_img)
            cv2.waitKey(1)
            return

        rgb_frame = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2RGB)
        start_time = rospy.Time.now()
        results = self.model.predict(rgb_frame, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False, device=self.device)[0]
        inference_time = (rospy.Time.now() - start_time).to_sec()

        self.inference_times.append(inference_time)
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            avg = sum(self.inference_times[-30:]) / 30.0
            rospy.loginfo(f"[YOLO Vision] FPS: {1.0 / avg:.1f}")

        # First, detect color buttons
        annotated, red_buttons, blue_buttons = self.detect_color_buttons(self.rgb_image.copy())
        
        # Reset handle centers
        self.latest_handle_center = None
        self.latest_cup_center = None
        
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()

            for box, cls, conf in zip(boxes, classes, confs):
                label = self.text_prompts[int(cls)]
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                if label == "black handle":
                    self.latest_handle_center = (cx, cy)
                    # Draw a crosshair marker at the center of the handle
                    cv2.drawMarker(
                        annotated,
                        (cx, cy),
                        color=(0, 0, 255),
                        markerType=cv2.MARKER_CROSS,
                        markerSize=20,
                        thickness=2
                    )
                    
                    # If depth data available, calculate distance information
                    if self.raw_depth is not None:
                        depth_val = self.raw_depth[cy, cx]
                        if depth_val > 0:
                            # Calculate distance to handle
                            fx, fy = 615.0, 615.0
                            cx_cam, cy_cam = 320.0, 240.0
                            X = (cx - cx_cam) * depth_val / 1000.0 / fx
                            Y = (cy - cy_cam) * depth_val / 1000.0 / fy
                            Z = depth_val / 1000.0
                            
                            # Display distance information
                            cv2.putText(annotated, f"Depth: {depth_val/1000.0:.2f}m", (x1, y2 + 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                elif label == "cup":
                    self.latest_cup_center = (cx, cy)
                    # Draw a blue crosshair marker at the center of the cup
                    cv2.drawMarker(
                        annotated,
                        (cx, cy),
                        color=(255, 0, 0),
                        markerType=cv2.MARKER_CROSS,
                        markerSize=20,
                        thickness=2
                    )
                    
                    # If depth data available, calculate distance information
                    if self.raw_depth is not None:
                        depth_val = self.raw_depth[cy, cx]
                        if depth_val > 0:
                            cv2.putText(annotated, f"Depth: {depth_val/1000.0:.2f}m", (x1, y2 + 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Draw the bounding box and label for all detected objects
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"{label} {conf:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Add a crosshair at the center of the image
        height, width = annotated.shape[:2]
        center_x, center_y = width // 2, height // 2
        cv2.drawMarker(annotated, (center_x, center_y), 
                    color=(0, 255, 255), markerType=cv2.MARKER_CROSS, 
                    markerSize=20, thickness=2)

        # Add FPS information
        cv2.putText(annotated, f"FPS: {1.0 / inference_time:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Add end-effector position information to bottom left
        if hasattr(self, 'pose'):
            cv2.putText(annotated, f"EE x: {self.pose[0]:.3f}m", (10, 400),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(annotated, f"EE y: {self.pose[1]:.3f}m", (10, 430),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(annotated, f"EE z: {self.pose[2]:.3f}m", (10, 460),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Display the annotated RGB image
        cv2.imshow("Object Detection", annotated)
        
        # Display the depth image if available
        if self.depth_image is not None:
            cv2.imshow("Depth Stream (0.1–3.0m)", self.depth_image)

        # Process key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            rospy.signal_shutdown("User exited")
        
        # You can add additional key handlers here if needed
        self.userDisplay.checkClose()


    def readObjects(self):
        """ read in goal object information"""

        #with open('/home/sarah/sonoma_ws/src/bci_jaco_matlab/src/vision/handle_pos.csv') as csvfile:
        with open('/home/hongyy/catkin_ws/src/bci_jaco_matlab_2004/src/vision/handle_pos.csv') as csvfile:
            C = csv.reader(csvfile)
            A = list(C)
            print(A)

        pos     = []
        self.goalObjs = list()

        # make goal object for each entry
        for idx, obj in enumerate(A):
            print(idx, obj)

            pos = list(map(float, obj[0:3]))
            ratio = float(obj[3])
            lr = float(obj[4])
            pos[1] = float(obj[5])

            self.goalObjs.append(GoalObject(pos, ratio, lr))

        self.numObjects = len(A)
        print("num obj: ", self.numObjects)
        self.possibleGoals = range(0, self.numObjects)

        # find nearest neighbor object
        for idx1, obj1 in enumerate(self.goalObjs):
            print("OBJ ", obj1.pos)
            for idx2, obj2 in enumerate(self.goalObjs):
                tmp =  obj2.pos - obj1.pos

                if np.linalg.norm([tmp[2], tmp[0]]) > 0:
                    ang = np.arctan2(tmp[2], -tmp[0])
                    ang = ang % (2*np.pi)
                    print("angle", idx2, ang)

                    if ang < np.pi/4:
                        self.goalObjs[idx1].nn[1] = idx2
                    elif ang < np.pi*3/4:
                        self.goalObjs[idx1].nn[5] = idx2
                    elif ang < np.pi*5/4:
                        self.goalObjs[idx1].nn[3] = idx2
                    elif ang < np.pi*7/4:
                        self.goalObjs[idx1].nn[6] = idx2
                    elif ang < np.pi*8/4:
                        self.goalObjs[idx1].nn[1] = idx2

        print(self.goalObjs[0].nn)
        print(self.goalObjs[1].nn)
        print(self.goalObjs[2].nn)

        if self.PreviewTarget:
            for i in range(0,self.numObjects):

                position    = self.goalObjs[i].pos
                # quaternion  = tf.transformations.quaternion_from_euler(self.goalObjs[i].eul[0], self.goalObjs[i].eul[1], self.goalObjs[i].eul[2],'rxyz')  
                quaternion = self.goalObjs[i].q
                result      = cartesian_pose_client(position, quaternion, self.prefix)
                
                position    = self.goalObjs[i].gpos
                # print(self.goalObjs[i].nn)
                # quaternion  = tf.transformations.quaternion_from_euler(self.wristStartX, self.wristStartY, self.wristStartZ,'rxyz')  
                result      = cartesian_pose_client(position, quaternion, self.prefix)

                input("Press Enter to continue")

                position    = self.goalObjs[i].ppos
                # quaternion  = tf.transformations.quaternion_from_euler(self.wristStartX, self.wristStartY, self.wristStartZ,'rxyz')  
                result      = cartesian_pose_client(position, quaternion, self.prefix)


    def removeGoal(self):
        """ removes goal object from list"""
        self.possibleGoals.remove(self.currentGoal)
        print("PG:", self.possibleGoals)
        self.numObjects = self.numObjects - 1
        
        if self.currentGoal == 1:
            self.goalObjs[0].nn[4] = 2
            self.goalObjs[2].nn[2] = 0

    def initializeBI(self):
        """"initialize Bayesian inference"""

        if self.numObjects == 3:  #TODO: change for generalized number of objects

            d = self.pDiag
            l = (1 - self.pDiag)/(2.0)
            self.P =np.array([[d,l,l], [l, d, l],  [l, l, d]])
            print(self.P)

        elif self.numObjects == 2:
            self.P =np.array([[0.9, 0.1], [0.1, 0.9]])
        elif self.numObjects == 1:
            self.P = [1]

        self.b = np.ones(self.numObjects)/self.numObjects
        self.theta_d = np.zeros(self.numObjects)
        self.theta_v = np.zeros(self.numObjects)
        self.goalSelected = 0
        self.startGrasp = 0
        self.objectGrasp = 0

    def updateBI_Cos(self):
        """ Update Bayesian inference"""

        k = 100
        B = 0.1
        v = np.array([self.uV[0], self.uV[2]])
        x = self.pose[0]
        y = self.pose[1]
        z = self.pose[2]

        X = np.array([x,z])
        # theta_d = np.zeros(self.numObjects)
        b_tmp = np.zeros(self.numObjects)

        d = np.zeros(self.numObjects)
        A1 = np.zeros(self.numObjects)
        C1 = np.zeros(self.numObjects)
        C2 = np.zeros(self.numObjects)
        C3 = np.zeros(self.numObjects)

        for i in range(0,self.numObjects):
        # for i in self.possibleGoals:
            k = self.possibleGoals[i]
            gx = self.goalObjs[k].pos[0]
            gz = self.goalObjs[k].pos[2]
            Xg = np.array([gx,gz])
            target_d = X - Xg
            d[i] = np.linalg.norm(target_d)


        min_dist = d.min()
        # print("min_dist:", min_dist)

        # print("Vel: ", v)
        for i in range(0,self.numObjects):
        # for i in range(1):
            k = self.possibleGoals[i]  
            gx = self.goalObjs[k].pos[0]
            gz = self.goalObjs[k].pos[2]
            Xg = np.array([gx,gz])
            theta_v = self.theta_v
            target_d = Xg - X
            # print("target_d ", target_d)
            # print("vel", v)
            # print("dot ", np.dot(target_d, v))
            thresh = 0.1

            # velocity heading         
            if np.linalg.norm(target_d) < .01:  # at target
                theta_v[i] = 0
            else:
                if np.linalg.norm(v) > self.slowThresh:
                    theta_v[i] = np.dot(target_d, v)/(np.linalg.norm(v)*np.linalg.norm(target_d))
                else:
                    # theta_v[i] = 0.95*self.theta_v[i]
                    # print("SLOW")
                    theta_v[i] = 1

            k = self.velK
            B = self.velB

            a1 = np.exp(B*k*theta_v[i]+1)
            a2 = 1


            if np.linalg.norm(v) < self.slowThresh:
                a1 = 1

            A1[i] = a1

            # distance to target
            if min_dist < 0.2:

                theta_d = self.theta_d
                k = self.distK
                B = self.distB
                theta_d[i] = np.linalg.norm(target_d)

                c1 = np.exp(-B*k*theta_d[i])
                c2 = 0

                if self.numObjects > 1:
                    for j in range(0,self.numObjects):
                        c2 = c2 + self.P[i,j]*self.b[j]
                        # c2 = 1
                else:
                    c2 = 1
            else:
                c1 = 0.5
                c2 = 0.5
                theta_d = np.zeros(self.numObjects)

            C1[i] = c1 
            C2[i] = c2
            c3 = 1

            if ((d[i] < .08) and (self.key == 2)):
                c3 = 0.8
            elif (d[i] < .08) and (self.key == 4):
                c3 = 1.2
            else:
                c3 = 1

            C3[i] = c3

            b_tmp[i] = a1*a2*c1*c2*c3

        self.theta_d = theta_d
        if self.numObjects > 1:
            self.b = b_tmp/sum(b_tmp)
        else:
            self.b = b_tmp/6

        thresh = self.beliefThresh

        s = np.array([b[0] for b in sorted(enumerate(self.b),key=lambda i:i[1], reverse=True)])

        if self.numObjects > 1:
            b_diff = self.b[s[0]] - self.b[s[1]]
        else:
            b_diff = self.b

        # print("BDIFF: ", b_diff)
        # print(b_diff)

        if b_diff > thresh:
            ind = self.b.argmax()
            self.currentGoal = self.possibleGoals[ind]
            gx = self.goalObjs[self.currentGoal ].pos[0]
            gz = self.goalObjs[self.currentGoal ].pos[2]
            Xg = np.array([gx,gz])
            self.goal_d = np.linalg.norm(Xg - X)
            self.goal_normVec = (Xg - X)/self.goal_d
            self.Assist = 1 

            print("Goal Assist: ", self.currentGoal, self.goalObjs[self.currentGoal].pos)

            if self.AssistMode == 1:
                self.blendedAssist()
                self.userDisplay.changeBGColor((0,0,255))
                print("ASSIST")
                # self.operationalMode = 5
                # self.setOperationMode()
                # self.assistStart = 1
            else:
                self.operationalMode = 5
                self.setOperationMode()
                self.assistStart = 1
                self.runningInput = np.zeros(self.runningInputBinNum)
                self.longInput = 0
                self.currentGoal = self.possibleGoals[ind]
                self.AssistLocked = 1
                self.lastAssistLockTime = datetime.datetime.now()
                self.runningInput = np.zeros(self.runningInputBinNum)
                self.switchAssist()
        else:
            self.userDisplay.changeBGColor((255,0,0))
            self.Assist = 0
        # print("Theta: ", theta_d)

    def switchAssist(self):
        """execute assistance for supervised autonomy with goal switching """
        # move to current goal

        x = self.pose[0]
        y = self.pose[1]
        z = self.pose[2]
        X = np.array([x,y,z])

        gx = self.goalObjs[self.currentGoal].pos[0]
        gy = self.goalObjs[self.currentGoal].pos[1]
        gz = self.goalObjs[self.currentGoal].pos[2]
        Xg = np.array([gx,gy,gz])
        self.goal_d = np.linalg.norm(Xg - X)
        self.goal_normVec = (Xg - X)/self.goal_d 

        alpha = 0

        if self.goal_d > 0.01:
            AssistVel = self.goal_normVec
            self.V[0] = (1-alpha)*AssistVel[0]*0.9*max(0.08, self.goal_d) 
            self.V[1] = (1-alpha)*AssistVel[1]*0.9*max(0.08, self.goal_d) 
            self.V[2] = (1-alpha)*AssistVel[2]*0.9*max(0.08, self.goal_d)
            self.goalReached = 0
        else:
            self.goalReached = 1
            self.V[0] = 0
            self.V[1] = 0
            self.V[2] = 0

        q_des = np.array(self.goalObjs[self.currentGoal].q)
        q_diff = np.linalg.norm(self.quaternion - q_des)

        if q_diff > 0.1:
            self.R[2] = -min(1., 2*np.abs(q_diff))
            self.rotReached = 0
        else:
            self.R[2] = 0
            self.rotReached = 1

        # print("Long Input: ", self.longInput)      

        if self.longInput == 7:  # select correct 
            self.goalSelected = 1

        elif self.longInput < 7:
            # print(self.goalObjs[self.currentGoal].nn)
            if self.goalObjs[self.currentGoal].nn[self.longInput] > -1:
                newGoal = self.goalObjs[self.currentGoal].nn[self.longInput]
                # print(newGoal)
                if newGoal in self.possibleGoals:
                    self.currentGoal = newGoal
                    self.runningInput = np.zeros(self.runningInputBinNum)

        if self.goalSelected == 1 and self.goalReached == 1 and self.rotReached:

            # open gripper
            if self.gripper == 1:
                self.setGripper(5000)
                self.gripper = 0

            self.startGrasp = 1
            self.graspState = 0


    def graspRoutine(self):
        openRoutine = self.goalObjs[self.currentGoal].openRoutine

        if self.graspState == 0:
            if openRoutine == 0:  # drawer
                # Original drawer code remains unchanged
                pos = self.goalObjs[self.currentGoal].SP
                grasp_pos = self.goalObjs[self.currentGoal].gpos
                open_pos = self.goalObjs[self.currentGoal].opos

                quaternion = self.quaternion
                print("QUAT", quaternion)
                result = cartesian_pose_client(grasp_pos, quaternion, self.prefix)
                print(grasp_pos)
                result = gripper_client([7400, 7400, 7400], self.prefix)
                print(open_pos)
                result = cartesian_pose_client(open_pos, quaternion, self.prefix)

                result = gripper_client([0, 0, 0], self.prefix)

                print(self.goalObjs[self.currentGoal].SP)
                position = self.goalObjs[self.currentGoal].SP[0]
                result = cartesian_pose_client(position, quaternion, self.prefix)

                position = self.goalObjs[self.currentGoal].SP[1]
                result = cartesian_pose_client(position, quaternion, self.prefix)

                quaternion = self.goalObjs[self.currentGoal].QP
                position = self.goalObjs[self.currentGoal].SP[2]
                result = cartesian_pose_client(position, quaternion, self.prefix)

                position = self.goalObjs[self.currentGoal].SP[3]
                result = cartesian_pose_client(position, quaternion, self.prefix)

                position = self.goalObjs[self.currentGoal].SP[4]
                result = cartesian_pose_client(position, quaternion, self.prefix)

            else:  # door
                # Dynamically set up the subpose sequence
                self.setup_open_sequence(self.currentGoal)
                
                # Execute the door opening sequence
                grasp_pos = self.goalObjs[self.currentGoal].gpos
                open_pos = self.goalObjs[self.currentGoal].opos
                quaternion = self.quaternion
                
                print("QUAT", quaternion)
                print("Starting door opening sequence...")
                
                # Move to grasp position
                print("Moving to grasp position:", grasp_pos)
                result = cartesian_pose_client(grasp_pos, quaternion, self.prefix)
                rospy.sleep(1.0)  # Wait to ensure position is reached
                
                # Close gripper to grasp the handle
                print("Closing gripper")
                result = gripper_client([7400, 7400, 7400], self.prefix)
                rospy.sleep(0.5)  # Wait for gripper to close
                
                # Move to open position (prepare for door opening motion)
                print("Moving to open position:", open_pos)
                result = cartesian_pose_client(open_pos, quaternion, self.prefix)
                rospy.sleep(1.0)
                
                # Execute the subpose sequence
                for i, position in enumerate(self.goalObjs[self.currentGoal].SP):
                    print(f"Moving to subpose {i}:", position)
                    result = cartesian_pose_client(position, quaternion, self.prefix)
                    rospy.sleep(1.0)  # Wait between movement steps
                    
                    # After the first subpose, open the gripper to release handle
                    if i == 0:
                        print("Opening gripper to release handle")
                        result = gripper_client([0, 0, 0], self.prefix)
                        rospy.sleep(0.5)
                
                print("Door opening sequence completed")
                
            self.graspState = 1

        elif self.graspState == 1:
            self.operationalMode = 0
            self.setOperationMode()
            self.assistStart = 0
            self.TaskMode = 2
    
    def setup_open_sequence(self, goal_index):
        """Set up the door opening sequence waypoints for a given goal"""
        print("Setting up door open sequence...")
        
        # Reference the goal object we're working with
        goal = self.goalObjs[goal_index]
        
        # Store original grasp and open positions
        grasp_pos = goal.gpos
        open_pos = goal.opos
        
        # Create subpose sequence
        sub_pos1 = [open_pos[0], open_pos[1], open_pos[2]]
        sub_pos1[1] += 0.01
        sub_pos1[0] -= 0.03
        
        sub_pos1a = sub_pos1[:]
        sub_pos1a[1] += 0.01
        
        sub_pos2 = sub_pos1a[:]
        sub_pos2[0] += 0.18
        
        sub_pos3 = sub_pos2[:]
        sub_pos3[1] -= 0.14
        
        sub_pos4 = sub_pos3[:]
        sub_pos4[0] -= 0.10
        
        sub_pos4a = sub_pos4[:]
        sub_pos4a[1] += 0.10
        
        sub_pos5 = sub_pos4a[:]
        sub_pos5[0] -= 0.05
        sub_pos5[1] += 0.05
        
        sub_pos5a = sub_pos5[:]
        sub_pos5a[1] -= 0.05
        
        sub_pos6 = sub_pos5[:]
        sub_pos6[0] -= 0.20
        
        sub_pos7 = sub_pos6[:]
        sub_pos7[0] += 0.10
        sub_pos7[1] -= 0.12
        sub_pos7[2] -= 0.02
        
        # Update the goal's SP (SubPose) sequence
        goal.SP = [sub_pos1, sub_pos1a, sub_pos2, sub_pos3, sub_pos4, 
                sub_pos4a, sub_pos5, sub_pos5a, sub_pos6, sub_pos7]
        
        print("Door open subposes setup complete.")

    def updateRunningSwitch(self, g):
        if self.switchLocked == 1:
            t = datetime.datetime.now()
            tdelta = (t - self.lastSwitchTime).seconds
            # print("TDELTA", tdelta)
            if tdelta >= self.switchLockTime:
                self.switchLocked = 0

        if self.switchLocked == 0:
            self.runningSwitch = np.insert(self.runningSwitch[1:], self.runningSwitch.size-1, g)

        self.switchCnt = np.count_nonzero(self.runningSwitch == self.switchInput)
        self.switchPercent = float(self.switchCnt)/float(self.runningSwitchBinNum)
        # print(self.runningSwitch)

    def updateRunningGrasp(self, g):
        self.runningGrasp = np.insert(self.runningGrasp[1:], self.runningGrasp.size-1, g)

        self.openCnt = np.count_nonzero(self.runningGrasp == 1)
        self.closeCnt = np.count_nonzero(self.runningGrasp == -1)

        self.openPercent = float(self.openCnt)/float(self.runningGraspBinNum)
        self.closePercent = float(self.closeCnt)/float(self.runningGraspBinNum)

    def updateRunningInput(self, input):
        # print(self.runningInput)
        # print("AL", self.AssistLocked)
        if self.AssistLocked == 1:
            self.longInput = 0
            t = datetime.datetime.now()
            tdelta = (t - self.lastAssistLockTime).seconds
            # print("TDELTA", tdelta)
            if tdelta >= self.assistLockTime:
                self.AssistLocked = 0

        if self.AssistLocked == 0:
            self.runningInput = np.insert(self.runningInput[1:], self.runningInput.size-1, input)

            vals,counts = np.unique(self.runningInput, return_counts=True)

            if np.max(counts) > (self.runningInputThresh*self.runningInputBinNum) and vals[np.argmax(counts)] != 0: 
                self.longInput = int(vals[np.argmax(counts)])
                self.AssistLocked = 1
                self.lastAssistLockTime = datetime.datetime.now()
                self.runningInput = np.zeros(self.runningInputBinNum)
            else:
                self.longInput = 0

    def reset(self):
        position    = self.home
        # print(position)
        self.operationalMode = self.operationalModeReset
        self.setOperationMode()
        self.userDisplay.screen.fill((0,0,0), (0,0,1000, 500))
        pg.display.flip()
        print("WRISTX", self.wristStartX)
        quaternion  = tf.transformations.quaternion_from_euler(self.wristStartX, self.wristStartY, self.wristStartZ,'rxyz')  
        print("Pos", position)
        print("QUAT", quaternion)
        result      = cartesian_pose_client(position, quaternion, self.prefix)
        self.runningGrasp = np.zeros(self.runningGraspBinNum)
        time.sleep(1)
        self.setGripper(0)
        self.gripper = 0
        time.sleep(.1)

        self.R[0] = 0
        self.R[1] = 0
        self.R[2] = 0
        self.V[0] = 0
        self.V[1] = 0
        self.V[2] = 0

        self.kv = self.kv_f
        self.ki = self.ki_f

        self.dist2D = 100
        self.agStep = 0
        self.t1 = 1
        self.t2 = 1
        self.initialApproach = 1
        self.graspInit= 0
        self.graspGo = 0
        self.graspGoal = 0
        self.EnableGoalSwitch = 1
        self.currentGoal = 0

        self.TaskMode = 1
        self.Assist = 0

        self.lockGripperClosed = 0
        
        if self.logOpen:
            self.file.close()
            self.logOpen = False
            self.inTrial = 0

        if self.logOpen2:
            self.file2.close()
            self.logOpen2 = False

    def stopRobot(self):
        self.setVelocity([0,0,0],[0,0,0])
        
    def setVelocity(self, V, R):
        duration_sec = 0.18
        p = 1.0;
        self.updateLogger()
        # self.checkInTargetGrasp()
        publishCatesianVelocityCommands([V[0], V[1], V[2], R[0], R[1], R[2]], duration_sec, self.prefix)

    def callbackPose(self, msg):
        self.pose = [msg.pose.position.x,msg.pose.position.y,msg.pose.position.z]

        self.quaternion = (
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w)
        self.euler = tf.transformations.euler_from_quaternion(self.quaternion, 'rxyz')

        # print('QUAT', self.quaternion)
        # q_desired = tf.transformations.quaternion_from_euler(3.1415/2, 3.1415/2, 0)
        # print('DES', q_desired)
        # q_diff = np.linalg.norm(self.quaternion - q_desired)
        # print('DIFF', q_diff )
        # print(self.pose)

        if self.logOpen2:

            self.updateLogger2()

        self.t0.pos  =  self.pose
        # print(self.euler)

    def callbackJoint(self, msg):
        self.jointAngles = (
            msg.joint1, msg.joint2,msg.joint3,msg.joint4,msg.joint5,msg.joint6,
            )
        # print(self.jointAngles[4] - self.jointAngles[3])

    def setMode(self, mode):
        self.mode = mode
        # self.home = [-0.2, -0.4, self.matlabPos[2]] 
        # if self.mode == 5 or self.mode > 7:
        self.home = [self.matlabPos[0] - 0.2, self.matlabPos[1] - .4, self.matlabPos[2]]
        self.reset()

    def startLogger(self, fn):
        self.file       = open(fn, "w")
        self.fileObj    = csv.writer(self.file)
        self.logOpen    = True
        self.logT       = time.time()

    def startLogger2(self, fn):
        self.file2       = open(fn, "w")
        self.fileObj2    = csv.writer(self.file2)
        self.logOpen2    = True
        self.logT2       = time.time()

    def updateLogger(self):
        bel = np.zeros(3)
        k = 0
        for i in self.possibleGoals:
            bel[i] = self.b[k]
            k = k + 1

        line = [time.time(), self.pose[0], self.pose[1], self.pose[2], self.V[0], self.V[1], self.V[2], self.fing, self.euler[0],self.euler[1], self.euler[2],self.operationalMode, self.key, self.R[0], self.R[1], self.R[2], bel[0], bel[1], bel[2], self.currentGoal, self.uV[0], self.uV[1], self.uV[2], self.Assist]
        self.fileObj.writerow(line)

    def updateLogger2(self):
        line = [time.time(), self.pose[0], self.pose[1], self.pose[2], self.quaternion[0],self.quaternion[1], self.quaternion[2],self.quaternion[3]]
        self.fileObj2.writerow(line)


    def setGripper(self,f):

        # workaround for getting stuck
        if (abs(self.jointAngles[4] - self.jointAngles[3]) < 5) | (abs(abs(self.jointAngles[4] - self.jointAngles[3]) - 360) < 5) | (abs(abs(self.jointAngles[4] - self.jointAngles[3]) - 180) < 5) :
            m = (self.jointAngles[4] + self.jointAngles[3])*0.5
            s = np.sign(self.jointAngles[4] - self.jointAngles[3]) 
            goalAng = (self.jointAngles[0],self.jointAngles[1], self.jointAngles[2], self.jointAngles[3] - s*5,self.jointAngles[4] + s*5, self.jointAngles[5],0.0)       
            result = joint_position_client(goalAng, self.prefix)
            time.sleep(.1)
            print("NUDGE")
        self.fing = max(f, self.fl)
        self.fing = min(f, self.fu)
        self.fing  = np.round(self.fing)

        self.fingers = [self.fing, self.fing, self.fing]
        result      = gripper_client(self.fingers, self.prefix)
        time.sleep(.1)

        self.goalMet = 1

    def switchGrasp(self):
        print("SWITCH GRASP")

        if self.gripper == 1:
            if self.lockGripperClosed == 0:
                #self.setGripper(self.gripperOpen)
                self.gripper = 0
            else:
                print("Gripper locked closed")

        elif self.gripper == 0:
            self.setGripper(self.gripperClose)
            self.gripper = 1

        self.runningSwitch = np.zeros(self.runningGraspBinNum)
        self.lastSwitchTime = datetime.datetime.now()
        self.switchLocked   = 1

    def inputToAction(self, input):
        self.updateRunningInput(input)
        self.modeswitch = 0
        self.key = input

        if self.TaskMode == 2 or self.AssistMode == 0:  # Transport 2 Place
                self.updateRunningSwitch(input)
                if self.switchPercent > self.runningSwitchThresh:
                    if self.UseModeSwitch:
                        self.switchModes()
                    else:
                        self.switchGrasp()

                        # if self.AssistMode > 0:
                        #     if self.numObjects > 2:
                        #         self.removeGoal()
                        #         self.TaskMode = 1
                        #         self.initializeBI()

        if self.operationalMode == 0 or self.operationalMode == 2:
            self.inputToXYZVel(input)
        elif self.operationalMode == 1:
            self.inputToWristGrasp(input)

        if self.TaskMode == 1: # ReachTo Grasp
            if self.startGrasp == 1:
                self.graspRoutine()
            else:
                if self.AssistMode ==1:     # blended
                    self.updateBI_Cos()
                elif self.AssistMode == 2:  # flexible autocomplete
                    if self.assistStart == 1:
                        self.switchAssist()
                    else:
                        self.updateBI_Cos()

    def inputToWristGrasp(self, input):
        self.key = input

        u = np.array([0., 0., 0.])
        u[1] = -(int(input == 1) - int(input == 3))
        u[0] = -(int(input == 2) - int(input == 4))
        u[2] = int(input == 5) - int(input == 6)
        u[0] = 0  #disable rotation for test

        if self.graspOrientation == 0:  #top-down
            if self.dampenUp and self.gripper == 0:
                u[2] = 0.2*int(input == 5) - int(input == 6)

            self.V[0] = 0
            self.V[1] = 0
            self.V[2] = self.kv*self.V[2] + self.ki*(u[2])
            self.R[2] = self.gkv*self.R[2] + -self.gki*(u[0])
        elif self.graspOrientation == 1:
            theta = self.euler[2]
            
            vy = np.cos(theta)*u[2]
            vx = -np.sin(theta)*u[2]

            self.V[0] =  self.kv*self.V[0] + self.ki*(vx)
            self.V[1] =  self.kv*self.V[1] + self.ki*(vy)
            self.V[2] = 0
            self.R[1] = self.gkv*self.R[1] + -self.gki*(u[0])


            if self.euler[2] <= self.latRotLim[0]:
                print("ROTATION LIMIT 1")
                self.R[1] = max(self.R[1],0)
            elif self.euler[2] >= self.latRotLim[1]:
                print("ROTATION LIMIT 2")
                self.R[1] = min(self.R[1],0)

            #("WORKSPACE LIMIT X")
        if self.pose[0] >= self.wu[0]:
            self.V[0] = min(self.V[0],0)
        
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
            self.FV = self.fv*self.FV + self.fi* u[1]
            self.fing = self.fing + self.FV*self.dt
            self.setGripper(self.fing)

        # Workspace limits
        if self.pose[0] <= self.wl[0]:
            self.V[0] = max(self.V[0],0)
            
        if self.pose[0] >= self.wu[0]:
            self.V[0] = min(self.V[0],0)

        if self.pose[1] <= self.wl[1]:
            self.V[1] = max(self.V[1],0)

        if self.pose[1] >= self.wu[1]:
            self.V[1] = min(self.V[1],0)

        if self.pose[2] <= self.wl[2]:
            self.V[2] = max(self.V[2],0)

        if self.pose[2] >= self.wu[2]:
            self.V[2] = min(self.V[2],0)

    def setOperationMode(self):
        if self.operationalMode == 1:
            self.runningSwitch = np.zeros(self.runningGraspBinNum)
            self.userDisplay.changeBGColor((0,0,255))
            print("Mode: Grasp")
            self.kv_s = 0.7    # damping coefficient
            self.ki_s = .01200   # mass coefficient 

            if self.autoCenterOverTarget == 1 and (self.dist2D < self.autoCenterDist):
                print("CENTER OVER TARGET")
                position    = [self.t1.pos[0], self.t1.pos[1], self.pose[2]]
                # print(position)
                result      = cartesian_pose_client(position, self.quaternion, self.prefix)
        elif self.operationalMode == 0:
            self.runningSwitch = np.zeros(self.runningGraspBinNum)
            self.userDisplay.changeBGColor((255,0,0))
            self.kv = self.kv_f
            self.ki = self.ki_f
            self.uV = [0,0,0]
            print("Mode: Translation")

        elif self.operationalMode == 2:
            self.runningSwitch = np.zeros(self.runningGraspBinNum)
            self.userDisplay.changeBGColor((0,255,0))
            self.kv = self.kv_s
            self.ki = self.ki_s
            print("Mode: Low Gain Translation")

        elif self.operationalMode == 3:
            self.runningSwitch = np.zeros(self.runningGraspBinNum)
            self.userDisplay.changeBGColor((255,255,0))
            self.kv = self.kv_s
            self.ki = self.ki_s
            print("Mode: Auto Grasp")

        elif self.operationalMode == 4:
            self.runningSwitch = np.zeros(self.runningGraspBinNum)
            self.userDisplay.changeBGColor((255,0,255))
            self.kv = self.kv_s
            self.ki = self.ki_s
            print("Mode: Auto Pose")

        elif self.operationalMode == 5:
            self.runningSwitch = np.zeros(self.runningGraspBinNum)
            self.userDisplay.changeBGColor((0,0,255))
            print("Mode: Assist")

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
        self.switchLocked   = 1
        self.modeswitch = self.operationalMode + 1

    def inputToXYZVel(self, input):
        self.key = input
        u = np.array([0., 0., 0., 0.])
        if self.view == 1:
            u[1] = -(int(input == 1) - int(input == 3))
            u[0] = -(int(input == 2) - int(input == 4))
            u[2] = int(input == 5) - int(input == 6)
            u[3] = int(input == 8) - int(input == 9)
        elif self.view==2:
            u[1] = -(int(input == 3) - int(input == 1))
            u[0] = -(int(input == 4) - int(input == 2))
            u[2] = int(input == 5) - int(input == 6)
            u[3] = int(input == 8) - int(input == 9)
        elif self.view==3:
            u[1] = -(int(input == 4) - int(input == 2))
            u[0] = -(int(input == 1) - int(input == 3))
            u[2] = int(input == 5) - int(input == 6)
            u[3] = int(input == 8) - int(input == 9)

        alpha = self.assistAlpha

        if self.UseSlowMode == 1:
            dist_from_base = np.sqrt(self.pose[0]*self.pose[0] + self.pose[1]*self.pose[1])

            if (dist_from_base > self.SlowDistanceThreshold):
                print("slow gain")
                self.uV[0] = self.kv_s*self.uV[0] + (1-alpha)*self.ki_s*u[0] 
                self.uV[1] = self.kv_s*self.uV[1] + (1-alpha)*self.ki_s*u[1]
                self.uV[2] = self.kv_s*self.uV[2] + self.ki_s*u[2] 
            else:
                print("standard gain")
                self.uV[0] = self.kv*self.uV[0] + (1-alpha)*self.ki*u[0] 
                self.uV[1] = self.kv*self.uV[1] + (1-alpha)*self.ki*u[1]
                self.uV[2] = self.kv*self.uV[2] + self.ki*u[2] 
        else:
            self.uV[0] = self.kv*self.uV[0] + (1-alpha)*self.ki*u[0] 
            self.uV[1] = self.kv*self.uV[1] + (1-alpha)*self.ki*u[1]
            self.uV[2] = self.kv*self.uV[2] + self.ki*u[2] 

        self.V = [self.uV[0], self.uV[1], self.uV[2]]
        # print(self.V)
        # Rotation X
        self.R[0] = self.gkv*self.R[0] + -self.gki*(u[3])
        # print(self.R[0])
        self.R[2] = 0

        if self.pose[0] <= self.wl[0]:
            self.V[0] = max(self.V[0],0)

            #("WORKSPACE LIMIT X")
        if self.pose[0] >= self.wu[0]:
            self.V[0] = min(self.V[0],0)
            #print("WORKSPACE LIMIT X")
        if self.pose[1] <= self.wl[1]:
            self.V[1] = max(self.V[1],0)
            #print("WORKSPACE LIMIT Y")
        if self.pose[1] >= self.wu[1]:
            self.V[1] = min(self.V[1],0)
            #print("WORKSPACE LIMIT Y")
        if self.pose[2] <= self.wl[2]:
            self.V[2] = max(self.V[2],0)
            print("WL", self.V)
            #print("WORKSPACE LIMIT Z")
        if self.pose[2] >= self.wu[2]:
            self.V[2] = min(self.V[2],0)
            #print("WORKSPACE LIMIT Z")

    def distance2D(self, p1,p2):
        a = np.array([p1[0], p1[1]])
        b = np.array([p2[0], p2[1]])
        d = np.linalg.norm(a - b)
        self.dist2D = d
        self.dist2Dvec = (a-b)/d
        self.distZ = p1[2] - p2[2]

    def saveTrialParams(self, fn):
        f      = open(fn, "w")
        fo   = csv.writer(f)

        line = [self.goalObjs[0].pos, self.goalObjs[1].pos, self.goalObjs[2].pos, self.AssistMode, self.AssistFn]
        fo.writerow(line)
        f.close()
