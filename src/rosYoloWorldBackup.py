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
from robot_control_modules import cartesian_pose_client
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
        self.text_prompts = ["door handle", "metal handle", "vertical handle", "vertial bar", "door knob", "black handle", 'Cup', 'Blue Plate', 'Red Plate']
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
            self.ee_position[1] + Y_remap + 0.12,  # safety offset
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
                rospy.loginfo(f"FPS: {1.0 / avg:.1f}")

            annotated = self.rgb_image.copy()
            self.latest_handle_center = None

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

                    if label == "black handle" and self.raw_depth is not None:
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
            cv2.imshow("YOLO World: Handle Detection", annotated)
            if self.depth_image is not None:
                cv2.imshow("Depth Stream (0.1–3.0m)", self.depth_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                rospy.signal_shutdown("User exited")
            elif key == ord("m") and self.latest_handle_center is not None:
                u, v = self.latest_handle_center
                depth_val = self.raw_depth[v, u]
                rospy.loginfo(f"Moving to handle at ({u}, {v}) depth: {depth_val} mm")
                self.move_to_target(u, v, depth_val)

            rate.sleep()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    YOLOWorldHandleDetector().run()
