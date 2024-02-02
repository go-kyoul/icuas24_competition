#-*- coding:utf-8 -*-
from cv_bridge import CvBridge
from Fruit_detector import Fruit_detector
import assosiation
from assosiation import FruitTracker
import message_filters

from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
import rospy
from std_msgs.msg import Bool, Int32MultiArray
import numpy as np



class RosInterface:
  def __init__(self):
    self.fruits_to_show = []
    self.cv_bridge = CvBridge()

    self.fruit_detector = Fruit_detector()
    self.static_fruit_tracker = FruitTracker()

    self.rgb_sub = message_filters.Subscriber('/red/camera/color/image_raw', Image)
    self.depth_sub = message_filters.Subscriber('/red/camera/depth/image_raw', Image)
    self.odom_sub = message_filters.Subscriber('/red/odometry', Odometry)

    self.ts = message_filters.ApproximateTimeSynchronizer(
      [self.rgb_sub, self.depth_sub, self.odom_sub], 1, 0.05, allow_headerless=True)
    self.ts.registerCallback(self.detection_callback)
    
    # twtw
    self.target_viewpoint_info = None
    self.viewpoint_validity_sub = rospy.Subscriber('/icuas2024_target_viewpoint', Int32MultiArray, self.validity_callback)


  def validity_callback(self, msg):
    target_viewpoint, valid = msg.data
    valid = valid == 1
    self.target_viewpoint_info = target_viewpoint, valid

  def detection_callback(self, rgb_msg, depth_msg, odom):
    if not self.target_viewpoint_info[1]:
      return
    
    if self.target_viewpoint_info[0] == assosiation.VIEWPOINT_BEFORE_START:
      return
    elif self.target_viewpoint_info[0] == assosiation.VIEWPOINT_AFTER_FINISH:
      plants_infos = self.static_fruit_tracker.clean_fruits()
      # print clustered
      for plants_info in plants_infos:
        print(plants_info[0], plants_info[1])

    # images
    rgb_image = self.cv_bridge.imgmsg_to_cv2(rgb_msg, "bgr8").copy()
    depth_image = self.cv_bridge.imgmsg_to_cv2(depth_msg, "passthrough").copy()
    
    # depth    
    depth_image[depth_image==np.nan] = 0
    np.clip(depth_image, 0, 10, depth_image)
    
    # detect fruits
    fruits, mask_debug, bboxes_drawn, bboxes_drawn2 = \
      self.fruit_detector.detect(rgb_image, depth_image, odom, ['pepper'])
    
    if len(fruits) > 0:
      self.static_fruit_tracker.update_tracked_fruits(fruits, self.target_viewpoint_info)

if __name__ == '__main__':
  rospy.init_node('fruit_detector')
  r = RosInterface()
  rospy.spin()