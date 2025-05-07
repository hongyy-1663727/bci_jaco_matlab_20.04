#!/usr/bin/env python3
"""
ROS node: yolo_world_handle_detector.py
Detects cabinet handles using YOLO World zero-shot object detection.
- Displays bounding boxes for text-prompted objects (e.g., "cabinet handle")
- RGB from /camera/color/image_raw
- Depth from /camera/aligned_depth_to_color/image_raw (0.1–3.0 m color mapped)
- Press 'm' to move robot end-effector to the center of the 'black handle' box (MoveIt required)
- Displays the distance from end-effector to 'metal handle' if detected

Requires:
- pip install ultralytics opencv-python numpy torch torchvision
- CPU/GPU support (YOLO World is much faster than OWL-ViT)
"""

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
from ultralytics import YOLOWorld
from geometry_msgs.msg import PoseStamped
import tf2_ros
import tf2_geometry_msgs
from moveit_commander import MoveGroupCommander, roscpp_initialize
from robot_control_modules import cartesian_pose_client, gripper_client
from kinova_msgs.msg import KinovaPose
import tf.transformations as tft
import tf


class YOLOWorldHandleDetector:
    def __init__(self):
        rospy.init_node("yolo_world_handle_detector", anonymous=True)
        roscpp_initialize([])

        self.bridge = CvBridge()
        self.rgb_image = None
        self.depth_image = None
        self.latest_handle_center = None
        self.raw_depth = None

        if torch.cuda.is_available():
            self.device = 0
            rospy.loginfo(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = 'cpu'
            rospy.logwarn("GPU not available, using CPU")

        self.model = YOLOWorld("yolov8x-worldv2.pt")
        self.model.to(self.device)
        self.text_prompts = ['black handle', 'cup']
        self.model.set_classes(self.text_prompts)
        rospy.loginfo("YOLO World model loaded")
        
        # Display a waiting message window
        
        waiting_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(waiting_img, "Waiting for camera data...", (50, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Camera Feed", waiting_img)
        
        cv2.waitKey(1)

        rospy.Subscriber("/camera/color/image_raw", Image, self.rgb_cb, queue_size=1, buff_size=2**24)
        rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_cb, queue_size=1, buff_size=2**24)
        
        self.ee_position = [0.0, 0.0, 0.0]
        self.ee_orientation_euler_deg = [0.0, 0.0, 0.0]


        rospy.Subscriber("/j2n6s300_driver/out/cartesian_command", KinovaPose, self.cartesian_pose_cb)


        self.conf_threshold = 0.2
        self.iou_threshold = 0.4
        self.inference_times = []
        self.frame_count = 0

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.arm = MoveGroupCommander("arm")
        
        # === After everything is initialized ===
        rospy.sleep(2.0)  # Allow time for move_group and topics to initialize

        rospy.loginfo("Homing end-effector to starting pose...")

        home_position = [-0.2, -0.4, 0.35]
        self.home_orientation = [0.7070904020014415, -0.0, 0.0, 0.7071231599922605]  # [x, y, z, w]

        prefix = "j2n6s300_"
        try:
            result = cartesian_pose_client(home_position, self.home_orientation, prefix)
            print("Pos", home_position)
            print("QUAT", self.home_orientation)
            if result:
                rospy.loginfo("Successfully moved to home position!")
            else:
                rospy.logwarn("Failed to move to home position!")
        except Exception as e:
            rospy.logwarn(f"Failed to home robot: {e}")
            


    def rgb_cb(self, msg):
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        
    def cartesian_pose_cb(self, msg):
        self.ee_position = [msg.X, msg.Y, msg.Z]
        self.ee_orientation_euler_deg = [msg.ThetaX, msg.ThetaY, msg.ThetaZ]
        
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
        cv2.imshow("Color Debug Mask", debug_mask)  # <---- THIS LINE
        
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


    def move_to_handle(self, u, v, depth_mm):
        # Just call your current `move_to_target` (or copy that part)
        # and then run the door opening sequence
        self.move_to_target(u, v, depth_mm)


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
            self.ee_position[1] + Y_remap + 0.17,  # safety offset
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
            self.grasp_position[1] + 0.12,
            self.grasp_position[2]
        ]

        # Use your original subpose sequence
        sub_pos1 = [self.opos[0], self.opos[1], self.opos[2]]
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

        self.SP = [sub_pos1, sub_pos1a, sub_pos2, sub_pos3, sub_pos4, sub_pos4a, sub_pos5, sub_pos5a, sub_pos6, sub_pos7]

        rospy.loginfo("Door open subposes setup complete.")



    def execute_open_sequence(self):
        rospy.loginfo("Executing door opening sequence...")

        try:
            # Move to grasp position
            result = cartesian_pose_client(self.gpos, self.grasp_orientation, "j2n6s300_")
            rospy.sleep(1.0)
            # Close gripper
            result = gripper_client([7400, 7400, 7400], "j2n6s300_")
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



    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if self.rgb_image is None:
                rate.sleep()
                continue

            rgb_frame = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2RGB)
            start_time = rospy.Time.now()
            results = self.model.predict(rgb_frame, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False, device=self.device)[0]
            inference_time = (rospy.Time.now() - start_time).to_sec()

            self.inference_times.append(inference_time)
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                avg = sum(self.inference_times[-30:]) / 30.0
                #rospy.loginfo(f"FPS: {1.0 / avg:.1f}")

            annotated, red_buttons, blue_buttons = self.detect_color_buttons(self.rgb_image.copy())

            
            self.latest_handle_center = None

            if results.boxes is not None and len(results.boxes) > 0:
                boxes = results.boxes.xyxy.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy()
                confs = results.boxes.conf.cpu().numpy()

                for box, cls, conf in zip(boxes, classes, confs):
                    label = self.text_prompts[int(cls)]
                    x1, y1, x2, y2 = map(int, box)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    if label == "cup":
                        self.latest_handle_center = (cx, cy)
                        self.latest_handle_label = "cup"

                        cv2.drawMarker(
                                annotated,
                                (cx, cy),
                                color=(255, 0, 0),  # Blue cross for cup
                                markerType=cv2.MARKER_CROSS,
                                markerSize=20,
                                thickness=2
                            )

                    if label == "cup" and self.raw_depth is not None:
                        depth_val = self.raw_depth[cy, cx]
                        
                        if depth_val > 0:
                            fx, fy = 615.0, 615.0
                            cx_cam, cy_cam = 320.0, 240.0
                            X = (cx - cx_cam) * depth_val / 1000.0 / fx
                            Y = (cy - cy_cam) * depth_val / 1000.0 / fy
                            Z = depth_val / 1000.0
                            
                            # === NEW: Remap Camera Frame (OpenCV) to EE Frame (Robot) ===
                            X_remap = -X          # Forward
                            Y_remap = -Z          # Left-Right
                            Z_remap = -Y         # Up-Down inverted

                            


                            pose_cam = PoseStamped()
                            pose_cam.header.stamp = rospy.Time.now()
                            pose_cam.header.frame_id = "j2n6s300_end_effector"  # direct to EE frame
                            pose_cam.pose.position.x = X_remap
                            pose_cam.pose.position.y = Y_remap
                            pose_cam.pose.position.z = Z_remap
                            pose_cam.pose.orientation.w = 1.0
                            
                            # === VISUALLY MARK THE SAMPLE POINT ===
                            
                            try:
                                # Transform handle pose into end-effector frame
                                self.pose_ee = self.tf_buffer.transform(pose_cam, "j2n6s300_end_effector", rospy.Duration(0.2))
                                dx = self.pose_ee.pose.position.x
                                dy = self.pose_ee.pose.position.y
                                dz = self.pose_ee.pose.position.z
                                distance = np.sqrt(dx**2 + dy**2 + dz**2)

                                # DISPLAY on Image
                                cv2.putText(annotated, f"X to EE: {dx:.2f}m", (x1, y2 + 20),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                cv2.putText(annotated, f"Y to EE: {dy:.2f}m", (x1, y2 + 40),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                cv2.putText(annotated, f"Z to EE: {dz:.2f}m", (x1, y2 + 60),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                
                                cv2.putText(annotated, f"EE x: {self.ee_position[0]:.3f}m", (10, 400),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                                cv2.putText(annotated, f"EE y: {self.ee_position[1]:.3f}m", (10, 430),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                                cv2.putText(annotated, f"EE z: {self.ee_position[2]:.3f}m", (10, 460),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                            except Exception as e:
                                rospy.logwarn_once(f"TF to EE failed: {e}")

                    if label == "black handle":
                        self.latest_handle_center = (cx, cy)
                        self.latest_handle_label = "black handle"


                    if label == "black handle" and self.raw_depth is not None:
                        depth_val = self.raw_depth[cy, cx]
                        self.detected_label = "black handle"
                        if depth_val > 0:
                            fx, fy = 615.0, 615.0
                            cx_cam, cy_cam = 320.0, 240.0
                            X = (cx - cx_cam) * depth_val / 1000.0 / fx
                            Y = (cy - cy_cam) * depth_val / 1000.0 / fy
                            Z = depth_val / 1000.0
                            
                            # === NEW: Remap Camera Frame (OpenCV) to EE Frame (Robot) ===
                            X_remap = -X          # Forward
                            Y_remap = -Z          # Left-Right
                            Z_remap = -Y         # Up-Down inverted

                            


                            pose_cam = PoseStamped()
                            pose_cam.header.stamp = rospy.Time.now()
                            pose_cam.header.frame_id = "j2n6s300_end_effector"  # direct to EE frame
                            pose_cam.pose.position.x = X_remap
                            pose_cam.pose.position.y = Y_remap
                            pose_cam.pose.position.z = Z_remap
                            pose_cam.pose.orientation.w = 1.0
                            
                            # === VISUALLY MARK THE SAMPLE POINT ===
                            cv2.drawMarker(
                                annotated,
                                (cx, cy),
                                color=(0, 0, 255),
                                markerType=cv2.MARKER_CROSS,
                                markerSize=20,
                                thickness=2
                            )



                            try:
                                # Transform handle pose into end-effector frame
                                self.pose_ee = self.tf_buffer.transform(pose_cam, "j2n6s300_end_effector", rospy.Duration(0.2))
                                dx = self.pose_ee.pose.position.x
                                dy = self.pose_ee.pose.position.y
                                dz = self.pose_ee.pose.position.z
                                distance = np.sqrt(dx**2 + dy**2 + dz**2)

                                # DISPLAY on Image
                                cv2.putText(annotated, f"X to EE: {dx:.2f}m", (x1, y2 + 20),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                cv2.putText(annotated, f"Y to EE: {dy:.2f}m", (x1, y2 + 40),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                cv2.putText(annotated, f"Z to EE: {dz:.2f}m", (x1, y2 + 60),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                
                                cv2.putText(annotated, f"EE x: {self.ee_position[0]:.3f}m", (10, 400),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                                cv2.putText(annotated, f"EE y: {self.ee_position[1]:.3f}m", (10, 430),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                                cv2.putText(annotated, f"EE z: {self.ee_position[2]:.3f}m", (10, 460),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                            except Exception as e:
                                rospy.logwarn_once(f"TF to EE failed: {e}")
                                
                    


                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated, f"{label} {conf:.2f}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.putText(annotated, f"FPS: {1.0 / inference_time:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # === Draw crosshair at image center ===
            height, width = annotated.shape[:2]
            center_x, center_y = width // 2, height // 2
            cv2.drawMarker(annotated, (center_x, center_y), color=(0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

            cv2.imshow("YOLO World: Handle Detection", annotated)
            if self.depth_image is not None:
                cv2.imshow("Depth Stream (0.1–3.0m)", self.depth_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                rospy.signal_shutdown("User exited")
            elif key == ord("m"):
                fx, fy = 615.0, 615.0
                cx, cy = 320.0, 240.0
                ee_x, ee_y, ee_z = self.ee_position

                best_button_uv = None
                min_dist = float("inf")

                # Combine buttons into one list with label
                button_points = [(pt, "red button") for pt in red_buttons] + [(pt, "blue button") for pt in blue_buttons]

                for (u, v), label in button_points:
                    if u < 0 or v < 0 or u >= self.raw_depth.shape[1] or v >= self.raw_depth.shape[0]:
                        continue

                    depth = self.raw_depth[v, u]
                    if depth <= 0 or np.isnan(depth):
                        rospy.logwarn(f"{label} at ({u},{v}) has invalid depth")
                        continue

                    # Convert to meters and to EE frame
                    X = (u - cx) * depth / 1000.0 / fx
                    Y = (v - cy) * depth / 1000.0 / fy
                    Z = depth / 1000.0

                    X_remap = -X
                    Y_remap = -Z
                    Z_remap = -Y

                    x_world = ee_x + X_remap
                    y_world = ee_y + Y_remap
                    z_world = ee_z + Z_remap

                    dist = np.linalg.norm([x_world - ee_x, y_world - ee_y, z_world - ee_z])
                    rospy.loginfo(f"{label} at ({u},{v}) is {dist:.3f}m from EE")

                    if dist < min_dist:
                        min_dist = dist
                        best_button_uv = (u, v)

                # === Move to closest button ===
                if best_button_uv:
                    u, v = best_button_uv
                    depth = self.raw_depth[v, u]

                    X = (u - cx) * depth / 1000.0 / fx
                    Y = (v - cy) * depth / 1000.0 / fy
                    Z = depth / 1000.0

                    X_remap = -X
                    Y_remap = -Z
                    Z_remap = -Y - 0.03

                    target_pos = [ee_x + X_remap, ee_y + Y_remap, ee_z + Z_remap]
                    target_ori = self.home_orientation

                    rospy.loginfo(f"Moving to button at ({u},{v}) → {target_pos}")

                    try:
                        result = cartesian_pose_client(target_pos, target_ori, "j2n6s300_")
                        if result:
                            rospy.loginfo("Moved to button.")
                        else:
                            rospy.logwarn("Move failed.")
                    except Exception as e:
                        rospy.logwarn(f"Exception during button move: {e}")

                elif self.latest_handle_center is not None:
                    # Fallback: handle or cup
                    u, v = self.latest_handle_center
                    depth_val = self.raw_depth[v, u]
                    rospy.loginfo(f"Moving to object ({self.latest_handle_label}) at ({u}, {v}), depth: {depth_val} mm")

                    if self.latest_handle_label == "cup":
                        self.move_to_cup(u, v, depth_val)
                    elif self.latest_handle_label == "black handle":
                        self.move_to_handle(u, v, depth_val)
                else:
                    rospy.logwarn("No buttons, cup, or handle detected — nothing to move to.")





            rate.sleep()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    YOLOWorldHandleDetector().run()
