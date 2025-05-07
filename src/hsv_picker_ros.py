#!/usr/bin/env python3
"""
ROS HSV Picker Tool
- Click anywhere on the color image window to print HSV value at that pixel
- Useful for tuning color detection ranges (e.g., red and blue buttons)
"""

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class HSVPicker:
    def __init__(self):
        rospy.init_node("hsv_picker", anonymous=True)
        self.bridge = CvBridge()
        self.rgb_image = None

        rospy.Subscriber("/camera/color/image_raw", Image, self.rgb_callback)

        cv2.namedWindow("HSV Picker")
        cv2.setMouseCallback("HSV Picker", self.on_mouse_click)

    def rgb_callback(self, msg):
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def on_mouse_click(self, event, x, y, flags, param):
        if self.rgb_image is None:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            hsv_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2HSV)
            pixel_hsv = hsv_image[y, x]
            h, s, v = pixel_hsv
            print(f"[HSV] @ ({x}, {y}): H={h}, S={s}, V={v}")

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.rgb_image is not None:
                display = self.rgb_image.copy()
                cv2.imshow("HSV Picker", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            rate.sleep()

        cv2.destroyAllWindows()

if __name__ == "__main__":
    picker = HSVPicker()
    picker.run()
