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
from kinova_msgs.msg import JointAngles, KinovaPose
import socket
import torch
import cv2
from ultralytics import YOLOWorld
from sensor_msgs.msg import Image
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped


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
        
        self.red_centers = []
        self.blue_centers = []
        
        # --- Vision (YOLO handle detection) ---
        self.bridge = CvBridge()
        self.rgb_image = None
        self.depth_image = None
        self.raw_depth = None
        self.latest_handle_center = None
        
        g1 = GoalObject([0.1, 0.2, 0.3], ratio=1.5, lr=0.5)
        g2 = GoalObject([0.2, 0.1, 0.4], ratio=1.2, lr=0.6)
        g3 = GoalObject([0.0, 0.3, 0.2], ratio=2.1, lr=0.4)
        
        self.home_orientation = [0.7070904020014415, -0.0, 0.0, 0.7071231599922605]  # [x, y, z, w]

        self.goalObjs = [g1, g2, g3]

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
        
        rospy.Subscriber("/j2n6s300_driver/out/cartesian_command", KinovaPose, self.cartesian_pose_cb)
        rospy.Subscriber("/camera/color/image_raw", Image, self.rgb_cb, queue_size=1, buff_size=2**24)
        rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_cb, queue_size=1, buff_size=2**24)

        self.conf_threshold = 0.25
        self.iou_threshold = 0.4
        self.inference_times = []
        self.frame_count = 0

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # right after setting up tf_listener
        self.ee_position = [0.0, 0.0, 0.0]
        self.ee_orientation_euler_deg = [0.0, 0.0, 0.0]


        self.last_rgb = rospy.Time.now()


        self.prefix     = 'j2n6s300_'
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.connect(('127.0.0.1', 43210))

        self.home       = [-0.20,-0.40, 0.2]
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
        self.wristStartY = 0
        # self.wristStartZ = 0.5*3.1415
        self.wristStartZ = 0.2

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
        
        self.controlCondition = 0
        # last time we flipped controlCondition
        self.last_toggle_time = rospy.Time(0)
        # minimum time between toggles
        self.toggle_debounce = rospy.Duration(0.5)  # 500 ms

        self.logOpen2 = 0
        subPose     = rospy.Subscriber('/j2n6s300_driver/out/tool_pose', PoseStamped, self.callbackPose)
        subJoint    = rospy.Subscriber('/j2n6s300_driver/out/joint_angles', JointAngles, self.callbackJoint)
        self.reset() 
        
    def cartesian_pose_cb(self, msg):
        self.ee_position = [msg.X, msg.Y, msg.Z]
        self.ee_orientation_euler_deg = [msg.ThetaX, msg.ThetaY, msg.ThetaZ]
        
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

        for mask, color, label, centers in [
                (red_mask, (0, 0, 255), "red button", red_centers),
                (blue_mask, (255, 0, 0), "blue button", blue_centers),
            ]:
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in cnts:
                    if cv2.contourArea(cnt) > 300:
                        x, y, w, h = cv2.boundingRect(cnt)
                        cx, cy = x + w // 2, y + h // 2
                        centers.append((cx, cy))
                        cv2.rectangle(image_bgr, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(image_bgr, label, (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        # --- NEW: Calculate 3D relative to EE ---
                        if self.raw_depth is not None:
                            depth_val = self.raw_depth[cy, cx]
                            if depth_val > 0:
                                fx, fy = 615.0, 615.0
                                cx_cam, cy_cam = 320.0, 240.0
                                X_cam = (cx - cx_cam) * depth_val / 1000.0 / fx
                                Y_cam = (cy - cy_cam) * depth_val / 1000.0 / fy
                                Z_cam = depth_val / 1000.0

                                # Remap to EE-relative frame
                                X_remap = -X_cam
                                Y_remap = -Z_cam
                                Z_remap = -Y_cam

                                pose_cam = PoseStamped()
                                pose_cam.header.stamp = rospy.Time.now()
                                pose_cam.header.frame_id = "j2n6s300_end_effector"
                                pose_cam.pose.position.x = X_remap
                                pose_cam.pose.position.y = Y_remap
                                pose_cam.pose.position.z = Z_remap
                                pose_cam.pose.orientation.w = 1.0

                                try:
                                    pose_ee = self.tf_buffer.transform(
                                        pose_cam,
                                        "j2n6s300_end_effector",
                                        rospy.Duration(0.2)
                                    )

                                    dx = pose_ee.pose.position.x
                                    dy = pose_ee.pose.position.y
                                    dz = pose_ee.pose.position.z

                                    # Display x, y, z rel to EE on image
                                    cv2.putText(image_bgr, f"dx: {dx:.2f}m", (x, y + h + 20),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                                    cv2.putText(image_bgr, f"dy: {dy:.2f}m", (x, y + h + 40),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                                    cv2.putText(image_bgr, f"dz: {dz:.2f}m", (x, y + h + 60),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                                except Exception as e:
                                    rospy.logwarn(f"[TF Error] {label}: {e}")

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
            #rospy.loginfo(f"[YOLO Vision] FPS: {1.0 / avg:.1f}")

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
                    # Display distance from image center (crosshair) to EE in camera frame (approx)
                    if hasattr(self, "pose_ee"):
                        dx = self.pose_ee.pose.position.x
                        dy = self.pose_ee.pose.position.y
                        dz = self.pose_ee.pose.position.z

                        cv2.putText(annotated, f"dx: {dx:.3f} m", (10, 320),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        cv2.putText(annotated, f"dy: {dy:.3f} m", (10, 350),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        cv2.putText(annotated, f"dz: {dz:.3f} m", (10, 380),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    
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
            cv2.putText(annotated, f"EE x: {self.ee_position[0]:.3f}m", (10, 400),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(annotated, f"EE y: {self.ee_position[1]:.3f}m", (10, 430),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(annotated, f"EE z: {self.ee_position[2]:.3f}m", (10, 460),
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
        self.wristStartZ = 0.2
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
            

    def setVelocity(self, V, R):
        duration_sec = 0.18
        p = 1.0;
        self.updateLogger()
        # self.checkInTargetGrasp()
        publishCatesianVelocityCommands([V[0], V[1], V[2], R[0], R[1], R[2]], duration_sec, self.prefix)


    def setMode(self, mode):
        self.mode = mode
        # self.home = [-0.2, -0.4, self.matlabPos[2]] 
        # if self.mode == 5 or self.mode > 7:
        #self.home = [self.matlabPos[0] - 0.2, self.matlabPos[1] - .4, self.matlabPos[2]]
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
        

        line = [time.time(), self.pose[0], self.pose[1], self.pose[2], self.V[0], self.V[1], self.V[2], self.fing, self.euler[0],self.euler[1], self.euler[2],self.operationalMode, self.key, self.R[0], self.R[1], self.R[2], bel[0], bel[1], bel[2], self.currentGoal, self.uV[0], self.uV[1], self.uV[2], self.Assist]
        self.fileObj.writerow(line)

    def updateLogger2(self):
        line = [time.time(), self.pose[0], self.pose[1], self.pose[2], self.quaternion[0],self.quaternion[1], self.quaternion[2],self.quaternion[3]]
        self.fileObj2.writerow(line)
        
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
        self.ee_position = list(self.pose)
        self.ee_orientation_euler_deg = [np.degrees(angle) for angle in self.euler]
        # print(self.euler)

    def callbackJoint(self, msg):
        self.jointAngles = (
            msg.joint1, msg.joint2,msg.joint3,msg.joint4,msg.joint5,msg.joint6,
            )
        # print(self.jointAngles[4] - self.jointAngles[3])


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
        
        
    def move_to_closest_visible_object(self):
        if self.rgb_image is None or self.raw_depth is None:
            rospy.logwarn("No image or depth data available")
            return
        
        # Ensure red/blue button detection is up to date
        _, self.red_centers, self.blue_centers = self.detect_color_buttons(self.rgb_image.copy())

        img_h, img_w = self.rgb_image.shape[:2]
        cx, cy = img_w // 2, img_h // 2

        # Collect candidates
        candidates = []

        if self.latest_handle_center:
            candidates.append(("black handle", self.latest_handle_center))
        if self.latest_cup_center:
            candidates.append(("cup", self.latest_cup_center))
        for r in self.red_centers:
            candidates.append(("red_button", r))
        for b in self.blue_centers:
            candidates.append(("blue_button", b))

        if not candidates:
            rospy.logwarn("No buttons, cup, or handle detected — nothing to move to.")
            return

        # Find closest to image center
        closest_obj = min(candidates, key=lambda c: np.linalg.norm(np.array(c[1]) - np.array([cx, cy])))
        label, (ux, uy) = closest_obj

        depth_val = self.raw_depth[uy, ux]
        if depth_val == 0:
            rospy.logwarn(f"{label} detected but no valid depth at that location")
            return
        
        # 2) camera→3D (meters)
        fx, fy = 615.0, 615.0
        cx_cam, cy_cam = 320.0, 240.0

        X_cam = (ux - cx_cam) * depth_val/1000.0 / fx
        Y_cam = (uy - cy_cam) * depth_val/1000.0 / fy
        Z_cam = depth_val/1000.0
        
        # === NEW: Remap Camera Frame (OpenCV) to EE Frame (Robot) ===
        X_remap = -X_cam        # Forward
        Y_remap = -Z_cam        # Left-Right
        Z_remap = -Y_cam        # Up-Down inverted
        
        # 3) build PoseStamped in camera frame
        pose_cam = PoseStamped()
        pose_cam.header.stamp = rospy.Time.now()
        pose_cam.header.frame_id = "j2n6s300_end_effector"        # ← your actual camera TF
        pose_cam.pose.position.x = X_remap
        pose_cam.pose.position.y = Y_remap
        pose_cam.pose.position.z = Z_remap
        pose_cam.pose.orientation.w = 1.0
        
        # 4) single TF into EE frame
        try:
            self.pose_ee = self.tf_buffer.transform(
                pose_cam,
                "j2n6s300_end_effector",     # ← your EE TF frame
                rospy.Duration(0.2)
            )
        except (tf2_ros.LookupException,
                tf2_ros.ExtrapolationException,
                tf2_ros.ConnectivityException) as e:
            rospy.logwarn(f"TF to EE failed: {e}")
            return
        
        # 5) read remapped offsets from EE pose
        dx = self.pose_ee.pose.position.x
        dy = self.pose_ee.pose.position.y
        dz = self.pose_ee.pose.position.z


        rospy.loginfo(f"Object {label} is at EE offset ({dx:.3f}, {dy:.3f}, {dz:.3f})")
        
        print(label)


        # Dispatch to appropriate routine
        if label == "black handle":
            rospy.loginfo("Initiating black handle grasp routine")
            self.move_to_target(ux, uy, depth_val)

        elif label == "cup":
            rospy.loginfo("Initiating cup grasp routine")
            self.move_to_cup(ux, uy, depth_val)

        elif label in ["red_button", "blue_button"]:
            rospy.loginfo(f"Moving directly to {label}")

            target_pos = [
                self.ee_position[0] + dx - 0.02,
                self.ee_position[1] + dy + 0.08,
                self.ee_position[2] + dz
            ]

            target_quat = tf.transformations.quaternion_from_euler(
                self.wristStartX, self.wristStartY, self.wristStartZ, 'rxyz'
            )

            try:
                result = cartesian_pose_client(target_pos, target_quat, self.prefix)
                if result:
                    rospy.loginfo(f"Moved to {label}.")
                else:
                    rospy.logwarn(f"Move to {label} failed.")
            except Exception as e:
                rospy.logwarn(f"Exception while moving to {label}: {e}")


    def move_to_cup(self, u, v, depth_mm):
            if depth_mm <= 0 or np.isnan(depth_mm):
                rospy.logwarn("Invalid depth for cup")
                return

            fx, fy = 615.0, 615.0
            cx, cy = 320.0, 240.0

            X_cam = (u - cx) * depth_mm / 1000.0 / fx
            Y_cam = (v - cy) * depth_mm / 1000.0 / fy
            Z_cam = depth_mm / 1000.0

            # Use TF pose (already calculated)
            
            X_remap = self.pose_ee.pose.position.x
            Y_remap = self.pose_ee.pose.position.y
            Z_remap = self.pose_ee.pose.position.z

            target_position = [
                self.ee_position[0] + X_remap + 0.05,
                self.ee_position[1] + Y_remap + 0.08,
                self.ee_position[2] + Z_remap + 0.08
            ]

            target_orientation = self.home_orientation


            prefix = "j2n6s300_"

            rospy.loginfo("Moving to cup...")

            try:
                result = cartesian_pose_client(target_position, target_orientation, prefix)
                if result:
                    rospy.loginfo("Reached cup!")
                    rospy.sleep(1.0)
                    gripper_client([7400, 7400, 7400], prefix)  # Grasp
                    rospy.loginfo("Cup grasped!")
                else:
                    rospy.logwarn("Move to cup failed!")

            except Exception as e:
                rospy.logwarn(f"Failed to move to cup: {e}")
                
    def move_to_target(self, u, v, depth_mm):
        if depth_mm <= 0 or np.isnan(depth_mm):
            rospy.logwarn("Invalid depth for target")
            return

        fx, fy = 615.0, 615.0
        cx, cy = 320.0, 240.0

        # Camera frame 3D point
        X_cam = (u - cx) * depth_mm / 1000.0 / fx
        Y_cam = (v - cy) * depth_mm / 1000.0 / fy
        Z_cam = depth_mm / 1000.0
        
        # Remap to robot EE frame - KEEP CONSISTENT with main loop
        X_remap = self.pose_ee.pose.position.x        # Forward
        Y_remap = self.pose_ee.pose.position.y        # Left-Right
        Z_remap = self.pose_ee.pose.position.z        # Up-Down inverted

        # Calculate target position relative to current EE position
        target_position = [
            self.ee_position[0] + X_remap + 0.05,
            self.ee_position[1] + Y_remap + 0.18,  # safety offset
            self.ee_position[2] + Z_remap + 0.08
        ]

        thetaX_deg, thetaY_deg, thetaZ_deg = self.ee_orientation_euler_deg
        thetaX_rad, thetaY_rad, thetaZ_rad = np.deg2rad([thetaX_deg, thetaY_deg, thetaZ_deg])

        # Kinova uses ZYX order (Yaw-Pitch-Roll)
        quat = tf.transformations.quaternion_from_euler(thetaZ_rad, thetaY_rad, thetaX_rad, axes='rzyx')
        target_orientation = [quat[0], quat[1], quat[2], quat[3]]



        prefix = "j2n6s300_"
        rospy.loginfo(f"Moving EE to: {target_position}")
        
        try:
            result = cartesian_pose_client(target_position, self.home_orientation, prefix)
            if result:
                rospy.loginfo("Successfully moved toward handle!")
            else:
                rospy.logwarn("Move failed!")
        except Exception as e:
            rospy.logwarn(f"Failed to move: {e}")
            
            
        self.grasp_position = self.ee_position.copy()
        self.grasp_orientation = self.home_orientation.copy()

        rospy.loginfo(f"Grasp pose saved: Pos {self.grasp_position}, Ori {self.grasp_orientation}")

        # Now trigger door opening sequence
        self.setup_open_sequence()
        self.execute_open_sequence()

    def setup_open_sequence(self):
        rospy.loginfo("Setting up door open sequence...")

        # Grasp and Open positions
        self.gpos = [
            self.grasp_position[0],
            self.grasp_position[1] - 0.04,
            self.grasp_position[2]
        ]

        self.opos = [
            self.grasp_position[0],
            self.grasp_position[1] + 0.07,
            self.grasp_position[2]
        ]

        # Use your original subpose sequence
        sub_pos1 = [self.opos[0], self.opos[1], self.opos[2]]
        sub_pos1[1] += 0.01
        sub_pos1[0] -= 0.01

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

        self.SP = [sub_pos1, sub_pos1a, sub_pos2, sub_pos3, sub_pos4, sub_pos4a, sub_pos5, sub_pos5a, sub_pos6, sub_pos7]

        rospy.loginfo("Door open subposes setup complete.")


    def execute_open_sequence(self):
        rospy.loginfo("Executing door opening sequence...")

        try:
            # Move to grasp position
            result = cartesian_pose_client(self.gpos, self.grasp_orientation, "j2n6s300_")
            rospy.sleep(1.0)
            # Close gripper
            result = gripper_client([7400, 7400, 0], "j2n6s300_")
            rospy.sleep(0.5)

            # Move to open position
            result = cartesian_pose_client(self.opos, self.grasp_orientation, "j2n6s300_")
            rospy.sleep(1.0)

            # Subposes SP[0] to SP[9]
            for i, pose in enumerate(self.SP):
                result = cartesian_pose_client(pose, self.grasp_orientation, "j2n6s300_")
                rospy.sleep(1.0)

                if i == 0:
                    # After first SP, release gripper
                    result = gripper_client([0, 0, 0], "j2n6s300_")
                    rospy.sleep(0.5)

            rospy.loginfo("Door opening sequence finished.")

        except Exception as e:
            rospy.logwarn(f"Error during door open sequence: {e}")

    def inputToAction(self, input):
        self.updateRunningInput(input)
        self.modeswitch = 0
        self.key = input
        
        # 2) toggle controlCondition on a long press of “3”
        if self.longInput == 4:
            # flip between 0 and 1
            self.controlCondition = 1 - self.controlCondition
            rospy.loginfo(f"[INPUT] longInput==4: switched controlCondition to {self.controlCondition}")
            if self.controlCondition == 0:
                # red bottom half
                self.userDisplay.changeBGColor((255, 0, 0))
            else:
                # blue bottom half
                self.userDisplay.changeBGColor((0, 0, 255))
            # reset so we don’t re-trigger on the same long press
            self.longInput = 0
        
        if self.longInput == 2:
            print("Activate action sequence")
            self.move_to_closest_visible_object()
            self.return_home_without_gripper()


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

    def return_home_without_gripper(self):
        rospy.loginfo("Returning to home position without changing gripper state...")
        quat = tf.transformations.quaternion_from_euler(self.wristStartX, self.wristStartY, self.wristStartZ, 'rxyz')  
        result = cartesian_pose_client(self.home, quat, self.prefix)
        if result:
            rospy.loginfo("Returned to home position.")
        else:
            rospy.logwarn("Failed to return to home.")


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
        elif self.view==4:
            u[0] = int(input == 2) - int(input == 1)  # +x / -x
            u[2] = int(input == 4) - int(input == 3)  # +z / -z
            u[1] = 0  # No movement in y
            u[3] = int(input == 8) - int(input == 9)  # Optional rotation if still needed
        elif self.view == 5:
            if self.controlCondition == 0:
                u[0] = int(input == 3) - int(input == 1)  # +x / -x
            elif self.controlCondition == 1:
                u[2] = int(input == 3) - int(input == 1)  # +x / -x

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
