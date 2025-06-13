#!/usr/bin/env python3
import rospy
import cv2
import torch
import rospkg
import os

from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from vision_msgs.msg import BoundingBox2DArray
from cv_bridge import CvBridge, CvBridgeError

from ultralytics import FastSAM

class FastSAMNode:
    def __init__(self):
        rospy.init_node('fastsam_ros', anonymous=True)

        self.model_name = rospy.get_param('~model_name', 'FastSAM-s.pt')
        self.prompt_mode = rospy.get_param('~prompt_mode', 'everything')
        self.text_prompt = rospy.get_param('~text_prompt', 'a photo of a car')

        self.device = rospy.get_param('~device', 'cpu')
        self.imgsz = rospy.get_param('~imgsz', 1024)
        self.conf = rospy.get_param('~conf', 0.4)
        self.iou = rospy.get_param('~iou', 0.9)
        self.retina_masks = rospy.get_param('~retina_masks', True)

        self.rospack = rospkg.RosPack()
        self.model_path = self.get_model_path()
        if not self.model_path:
            rospy.logerr("Failed to find model file. Shutting down.")
            return

        rospy.loginfo(f"Loading FastSAM model from: {self.model_path}")
        self.model = FastSAM(self.model_path)
        rospy.loginfo("FastSAM model loaded successfully.")

        self.latest_image = None
        self.latest_image_msg = None
        self.bridge = CvBridge()
        
        self.vis_pub = rospy.Publisher('~visualization', Image, queue_size=1)
        self.mask_pub = rospy.Publisher('~segmentation_mask', Image, queue_size=1)

        self.image_sub = rospy.Subscriber('~image_in', Image, self.image_callback, queue_size=1, buff_size=2**24)

        if self.prompt_mode == 'point':
            self.point_sub = rospy.Subscriber('~point_prompt', PointStamped, self.point_prompt_callback, queue_size=1)
            rospy.loginfo("Point prompt mode enabled. Waiting for points on topic '~point_prompt'.")
        elif self.prompt_mode == 'bbox':
            self.bbox_sub = rospy.Subscriber('~bbox_prompt', BoundingBox2DArray, self.bbox_prompt_callback, queue_size=1)
            rospy.loginfo("BBox prompt mode enabled. Waiting for bboxes on topic '~bbox_prompt'.")
        
        rospy.loginfo(f"FastSAM node initialized in '{self.prompt_mode}' mode. Waiting for inputs...")

    def get_model_path(self):
        try:
            package_path = self.rospack.get_path('fastsam_ros')
            model_path = os.path.join(package_path, 'models', self.model_name)
            if os.path.exists(model_path):
                return model_path
            else:
                rospy.logerr(f"Model file not found at {model_path}")
                return None
        except rospkg.ResourceNotFound:
            rospy.logerr("Package 'fastsam_ros' not found.")
            return None

    def image_callback(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_image_msg = msg
            if self.prompt_mode in ['everything', 'text']:
                self.run_inference(self.latest_image)
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

    def point_prompt_callback(self, msg):
        if self.latest_image is None:
            rospy.logwarn("Received a point prompt, but no image has been received yet.")
            return
        
        point = [[int(msg.point.x), int(msg.point.y)]]
        label = [1]
        rospy.loginfo(f"Received point prompt: {point}")
        self.run_inference(self.latest_image, points=point, labels=label)
        
    def bbox_prompt_callback(self, msg):
        if self.latest_image is None:
            rospy.logwarn("Received a bbox prompt, but no image has been received yet.")
            return

        bboxes = []
        for bbox in msg.boxes:
            x1 = int(bbox.center.x - bbox.size_x / 2)
            y1 = int(bbox.center.y - bbox.size_y / 2)
            x2 = int(bbox.center.x + bbox.size_x / 2)
            y2 = int(bbox.center.y + bbox.size_y / 2)
            bboxes.append([x1, y1, x2, y2])

        if not bboxes:
            rospy.logwarn("Received an empty BoundingBox2DArray.")
            return

        rospy.loginfo(f"Received bbox prompt: {bboxes}")
        self.run_inference(self.latest_image, bboxes=bboxes)

    def run_inference(self, image, bboxes=None, points=None, labels=None):
        kwargs = {
            "source": image, "device": self.device, "retina_masks": self.retina_masks,
            "imgsz": self.imgsz, "conf": self.conf, "iou": self.iou, "verbose": False
        }

        current_mode = self.prompt_mode
        if bboxes is not None:
            kwargs['bboxes'] = bboxes
            current_mode = 'bbox'
        elif points is not None and labels is not None:
            kwargs['points'] = points
            kwargs['labels'] = labels
            current_mode = 'point'
        elif current_mode == 'text':
            kwargs['texts'] = self.text_prompt
        
        rospy.loginfo(f"Running inference with mode: '{current_mode}'")
        try:
            results = self.model(**kwargs)
            if results:
                self.publish_visualization(results)
                self.publish_segmentation_mask(results)
        except Exception as e:
            rospy.logerr(f"Failed to run FastSAM inference: {e}")

    def publish_visualization(self, results):
        """Generates and publishes the visualization image."""
        try:
            vis_image = results[0].plot() 
            vis_msg = self.bridge.cv2_to_imgmsg(vis_image, "bgr8")
            vis_msg.header = self.latest_image_msg.header
            self.vis_pub.publish(vis_msg)
        except Exception as e:
            rospy.logerr(f"Failed to publish visualization: {e}")

    def publish_segmentation_mask(self, results):
        try:
            masks = results[0].masks
            if masks is None or len(masks.data) == 0:
                rospy.logwarn("No masks generated in the results.")
                return

            combined_mask_tensor = torch.any(masks.data, dim=0).cpu()
            combined_mask_np = (combined_mask_tensor.numpy() * 255).astype("uint8")
            
            mask_msg = self.bridge.cv2_to_imgmsg(combined_mask_np, "mono8")
            mask_msg.header = self.latest_image_msg.header
            self.mask_pub.publish(mask_msg)
        except Exception as e:
            rospy.logerr(f"Failed to publish segmentation mask: {e}")

if __name__ == '__main__':
    try:
        node = FastSAMNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logfatal(f"An unhandled exception occurred in FastSAM node: {e}")
