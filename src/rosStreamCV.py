#!/usr/bin/env python3
"""
ROS node: owlvit_handle_detector.py
Detects cabinet handles using OWL-ViT zero-shot object detection.
- Displays bounding boxes for text-prompted objects (e.g., "cabinet handle")
- RGB from /camera/color/image_raw
- Depth from /camera/aligned_depth_to_color/image_raw (0.1–3.0 m color mapped)

Requires:
- pip install torch torchvision transformers timm opencv-python
- GPU recommended (OWL-ViT is slow on CPU)
"""

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import torch
from PIL import Image as PILImage

class OWLVitHandleDetector:
    def __init__(self):
        rospy.init_node("owlvit_handle_detector", anonymous=True)

        self.bridge = CvBridge()
        self.rgb_image = None
        self.depth_image = None

        # Check GPU availability with better feedback
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            rospy.loginfo(f"Using GPU: {gpu_name}")
            rospy.loginfo(f"CUDA Version: {torch.version.cuda}")
        else:
            self.device = torch.device("cpu")
            rospy.logwarn("GPU not available, using CPU (detection will be slow)")

        # Load OWL-ViT model and processor
        try:
            rospy.loginfo("Loading OWL-ViT model...")
            self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
            self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
            self.model.to(self.device)
            rospy.loginfo("Model loaded successfully")
        except Exception as e:
            rospy.logerr(f"Failed to load model: {str(e)}")
            rospy.signal_shutdown("Model loading failed")

        # Text prompt to detect
        self.text_prompts = [["black handle", "metal handle", "door handle"]]

        # Subscribers
        rospy.Subscriber("/camera/color/image_raw", Image, self.rgb_cb, queue_size=1, buff_size=2**24)
        rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_cb, queue_size=1, buff_size=2**24)
        
        # Add performance tracking
        self.inference_times = []

    def rgb_cb(self, msg):
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def depth_cb(self, msg):
        raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        depth_m = raw.astype(np.float32) / 1000.0
        depth_m[np.isnan(depth_m)] = 0.0

        min_d, max_d = 0.1, 3.0
        valid = (depth_m >= min_d) & (depth_m <= max_d)
        norm = np.zeros_like(depth_m, dtype=np.uint8)
        norm[valid] = np.clip(((depth_m[valid] - min_d) / (max_d - min_d)) * 255, 0, 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        depth_colored[~valid] = (0, 0, 0)
        self.depth_image = depth_colored

    def run(self):
        rate = rospy.Rate(1)  # OWL-ViT is slower, 1Hz is reasonable
        while not rospy.is_shutdown():
            if self.rgb_image is None:
                rate.sleep()
                continue

            pil_image = PILImage.fromarray(cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2RGB))
            inputs = self.processor(text=self.text_prompts, images=pil_image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            target_sizes = torch.Tensor([pil_image.size[::-1]]).to(self.device)  # (H, W)
            results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.1)[0]

            annotated = self.rgb_image.copy()
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [int(i) for i in box]
                cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(annotated, f"{self.text_prompts[0][label]} {score:.2f}", (box[0], box[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Show results
            cv2.imshow("OWL-ViT: Handle Detection", annotated)
            if self.depth_image is not None:
                cv2.imshow("Depth Stream (0.1–3.0m)", self.depth_image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                rospy.signal_shutdown("User exited")

            rate.sleep()

        cv2.destroyAllWindows()

if __name__ == "__main__":
    OWLVitHandleDetector().run()
