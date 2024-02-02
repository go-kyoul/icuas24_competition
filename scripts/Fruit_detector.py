#-*- coding:utf-8 -*-

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2
import numpy as np
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray, Marker
from tf.transformations import quaternion_matrix, quaternion_from_matrix

class InvalidDetectionChecker: # all parameters used for cutting detection should be here...
  def __init__(self):
    self.fruit_DA_dist_thresh = 0.2 # meter
    self.fruit_cluster_dist_thresh = 0.7 # meter
    self.max_depth_thresh = 5.0 # meter

    self.image_width   = 640
    self.image_height  = 480
    self.bbox_border = 10 # pixel
    self.min_wh = (7, 7) # width, height must be equal or exceed this size
    self.max_fruit_height_minus_drone_height = 0.5 # meter

  def is_fruit_height_valid(self, fruit_height, drone_height):
    return fruit_height - drone_height < self.max_fruit_height_minus_drone_height

  def is_bbox_valid(self, xywh):
    xy_ok = xywh[0] >= self.bbox_border and xywh[1] >= self.bbox_border and \
      xywh[0] + xywh[2] <= self.image_width - self.bbox_border and \
        xywh[1] + xywh[3] <= self.image_height - self.bbox_border
    wh_ok = xywh[2] >= self.min_wh[0] and xywh[3] >= self.min_wh[1]
    return xy_ok and wh_ok
   
  def is_DA_dist_valid(self, dist):
    return dist < self.fruit_DA_dist_thresh

  def is_clustering_dist_valid(self, dist):
    return dist < self.fruit_cluster_dist_thresh
  
  def is_max_depth_exceeded(self, depth):
    return depth > self.max_depth_thresh

invalid_detection_checker = InvalidDetectionChecker()

class Simple2dHSVDetector:
  def __init__(self):
    self.fruit_mask_applier = {
      'tomato':self.apply_tomato_mask, 
      'eggplant':self.apply_eggplant_mask, 
      'pepper': self.apply_pepper_mask
    }
    global invalid_detection_checker
  
  def drawbbox(self, rgb_image, bbox_list):
    for bbox in bbox_list:
      x, y, w, h = bbox
      cv2.rectangle(rgb_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    return rgb_image

  def get_bbox(self, rgb_image, class_name):
    mask = self.fruit_mask_applier[class_name](rgb_image)
    bboxes, nonzeros = self.get_bbox_from_mask(mask)
    return bboxes, nonzeros, mask

  @staticmethod
  def get_bbox_from_mask(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    nonzeros = []
    for contour in contours:
      x, y, w, h = cv2.boundingRect(contour)
      bbox = [x, y, w, h]
      if not invalid_detection_checker.is_bbox_valid(bbox):
        continue
      bboxes.append(bbox)
      
      nonzero = \
        cv2.countNonZero(mask[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]])
      nonzeros.append(nonzero)
    return bboxes, nonzeros
  
  def apply_eggplant_mask(self, rgb_image):
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

    # Define thresholds for HSV channels
    channel1_min = 0.720 * 179  # Hue value scaled to OpenCV range [0, 179]
    channel1_max = 0.728 * 179
    channel2_min = 0.804 * 255  # Saturation and Value values scaled to OpenCV range [0, 255]
    channel2_max = 0.820 * 255
    channel3_min = 0.922 * 255
    channel3_max = 1.000 * 255

    # Create mask based on thresholds
    mask = cv2.inRange(hsv_image, (channel1_min, channel2_min, channel3_min), (channel1_max, channel2_max, channel3_max))

    # opening
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return opening
    
  def apply_tomato_mask(self, rgb_image):
    # Convert RGB image to HSV color space
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

    # Define thresholds for HSV channels
    # For red color, we need to handle the wrap-around at the hue channel ends
    channel1_min1 = 0.998 * 179  # Upper range of red
    channel1_max1 = 179
    channel1_min2 = 0           # Lower range of red
    channel1_max2 = 0.000 * 179
    channel2_min = 0.692 * 255
    channel2_max = 0.820 * 255
    channel3_min = 0.561 * 255
    channel3_max = 1.000 * 255

    # Create mask for upper and lower red ranges
    mask1 = cv2.inRange(hsv_image, (channel1_min1, channel2_min, channel3_min), (channel1_max1, channel2_max, channel3_max))
    mask2 = cv2.inRange(hsv_image, (channel1_min2, channel2_min, channel3_min), (channel1_max2, channel2_max, channel3_max))

    # Combine masks for the full red range
    masked = cv2.bitwise_or(mask1, mask2)

    # opening
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # closed = cv2.morphologyEx(masked, cv2.MORPH_CLOSE, kernel, iterations=10)

    # erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    erosion_dst = cv2.erode(closed, kernel, iterations=10)

    return erosion_dst

  def apply_pepper_mask(self, rgb_image):
    # Convert RGB image to HSV color space
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

    # Define thresholds for HSV channels
    channel1_min = 0.137 * 179  # Hue value scaled to OpenCV range [0, 179]
    channel1_max = 0.167 * 179
    channel2_min = 0.692 * 255  # Saturation and Value values scaled to OpenCV range [0, 255]
    channel2_max = 0.816 * 255
    channel3_min = 0.561 * 255
    channel3_max = 1.000 * 255

    # Create mask based on thresholds
    masked = cv2.inRange(hsv_image, (channel1_min, channel2_min, channel3_min), (channel1_max, channel2_max, channel3_max))

    # closing
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(masked, cv2.MORPH_CLOSE, kernel)

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # closed = cv2.morphologyEx(masked, cv2.MORPH_CLOSE, kernel, iterations=10)

    # erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded = cv2.erode(closed, kernel, iterations=3)

    return masked
  
class DetectedFruit:
  XYZ_moving_average_window_size = 4
  def __init__(self, fruit_name, n_fruit, xyz, odom):
    self.fruit_name = fruit_name
    self.n_fruit = n_fruit
    self.detected_time = odom.header.stamp
    self.xyz = np.array(xyz)
    self.xyzs = [self.xyz]
    self.drone_position = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z])

  
  def update_xyz(self, xyz):
    self.xyzs.append(xyz) # this function assumpes xyz is consistent
    if len(self.xyzs) > self.XYZ_moving_average_window_size:
      self.xyzs.pop(0)
    self.xyz = np.mean(self.xyzs, axis=0)
  
  def is_valid(self):
    return len(self.xyzs ) >= self.XYZ_moving_average_window_size
 
class StaticFruitCounter:
  def __init__(self):
    self.fruits = []
    self.invalid_check_detector = InvalidDetectionChecker()

  def add_fruit(self, detected_fruit):
    new_fruit = True
    for f in self.fruits:
      if f.fruit_name == detected_fruit.fruit_name:
        # calc distance
        dist = np.linalg.norm(f.xyz - detected_fruit.xyz, 2)
        if self.invalid_check_detector.is_DA_dist_valid(dist):
          f.update_xyz(detected_fruit.xyz)
          new_fruit = False
          break
    if new_fruit:
      self.fruits.append(detected_fruit)
    
  def cluster_fruits(self):
    fruits = list(self.fruits)
    fruits_clustered = []
    centroids = []

    while len(fruits) > 0:
      f = fruits.pop(0)

      # assign f to cluster. If failed, create new cluster
      assigned = False
      for fc in fruits_clustered:
        # calc centroid
        centroid = np.mean([f.xyz for f in fc], axis=0)
        dist_to_centroid = np.linalg.norm(centroid - f.xyz)
        if self.invalid_check_detector.is_clustering_dist_valid(dist_to_centroid):
          fc.append(f)
          assigned = True
          break
      if not assigned:
        fruits_clustered.append([f])
    for fc in fruits_clustered:
      centroids.append(np.mean([f.xyz for f in fc], axis=0))

    n_fruits = []
    for fc in fruits_clustered:
      n_fruit = sum([f.n_fruit for f in fc])
      n_fruits.append(n_fruit)
      
    # print('--- fruits clustered ---')
    # for n_fruit, c in zip(n_fruits, centroids):
    #   print(n_fruit, c)
    
    return fruits_clustered, centroids, n_fruits

  def print_fruits(self):
    print('--- list of fruits ---')
    for f in self.fruits:
      print(f.fruit_name, f.xyz)
  

class Fruit_detector:
  def __init__(self):
    self.fx  = 381.36246688113556
    self.fy  = 381.36246688113556
    self.cx  = 320.5
    self.cy  = 240.5

    self.detector_2d = Simple2dHSVDetector()


    self.invalid_check_detector = InvalidDetectionChecker()
    
  def detect(self, rgb_image, depth_image, odom, class_names):
    rgb_image = rgb_image
    depth_image = depth_image
    
    # rotation matrix from quaternion
    qx = odom.pose.pose.orientation.x
    qy = odom.pose.pose.orientation.y
    qz = odom.pose.pose.orientation.z
    qw = odom.pose.pose.orientation.w
    R = np.array([
      [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
      [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
      [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
    ])

    tr_w2c = np.zeros((4, 4))
    tr_w2c[:3, :3] = R
    tr_w2c[-1, -1] = 1
    tr_w2c[0, -1] = odom.pose.pose.position.x
    tr_w2c[1, -1] = odom.pose.pose.position.y
    tr_w2c[2, -1] = odom.pose.pose.position.z
    tr_w2c[3, -1] = 1

    fruits = []
    for class_name in class_names:
      bbox_list, n_nonzeros, mask_debug = self.detector_2d.get_bbox(rgb_image, class_name)

      center_of_mass_list = []
      for bbox in bbox_list:
        center_of_mass = self.calculate_center_of_mass(bbox, depth_image)
        if np.isnan(center_of_mass).any():
          continue
        center_of_mass_list.append(center_of_mass)
      
      xyzs_world = []
      valid_n_nonzeros = []
      for idx in range(len(center_of_mass_list)):
        # object position represented in world coordinate
        cam_x, cam_y, cam_z = center_of_mass_list[idx]
        if self.invalid_check_detector.is_max_depth_exceeded(cam_z):
          continue # cut far object
        xyz_world = np.dot(tr_w2c, np.array([cam_z+0.24, -cam_x, -cam_y+0.05, 1]))
        if self.invalid_check_detector.is_fruit_height_valid(xyz_world[2], odom.pose.pose.position.z):
          continue
        # xyz = np.dot(tr_w2c, np.array([cam_z +0.2, -cam_x, -cam_y+0.05, 1]))
        xyzs_world.append(xyz_world[:3])
        valid_n_nonzeros.append(n_nonzeros[idx])

      if len(xyzs_world) == 0:
        continue

      # find min_nonzeros
      min_nonzeros = np.min(valid_n_nonzeros)

      # count number of fruits
      for idx in range(len(valid_n_nonzeros)):
        n_fruit = round(valid_n_nonzeros[idx]/(320))
        xyz_world = xyzs_world[idx]

        f = DetectedFruit(class_name, n_fruit, xyz_world, odom)
        fruits.append(f)
    
    # debug: bboxes_drawn
    bboxes_drawn = \
      self.detector_2d.drawbbox(rgb_image.copy(), bbox_list)
    bboxes_drawn = cv2.resize(bboxes_drawn, (640, 480))

    # debug: bbox_drawn on depth image
    depth_image_255 = (depth_image.copy()/10.0*255).astype(np.uint8)
    depth_image_255 = cv2.cvtColor(depth_image_255, cv2.COLOR_GRAY2BGR)
    depth_image_255 = cv2.resize(depth_image_255, (640, 480))
    bboxes_drawn2 = \
      self.detector_2d.drawbbox(depth_image_255.copy(), bbox_list)

    return fruits, mask_debug, bboxes_drawn, bboxes_drawn2
    
  def depth_to_point(self, depth, u, v):
    z = depth[v, u]  # 주의: NumPy 배열??? (???, ???) ???????? ??????????????????.
    x = (u - self.cx) * z / self.fx 
    y = (v - self.cy) * z / self.fy
    return x, y, z
  
  def calculate_center_of_mass(self, bbox, depth_image):
    x, y, w, h = bbox

    # Bounding box ?????? 모든 ?????? 좌표 ??????
    u, v = np.meshgrid(np.arange(x, x + w), np.arange(y, y + h))

    # depth_to_point ?????? ??????
    points = np.array(
      [self.depth_to_point(depth_image, u[i, j], v[i, j]) 
        for i in range(h) for j in range(w)]
    )

    # ignore nan and zero and inf points
    valid_points = points[~np.isnan(points).any(axis=1)]
    valid_points = valid_points[~np.isinf(valid_points).any(axis=1)]
    valid_points = valid_points[~(valid_points == 0).all(axis=1)]
    center_of_mass = np.mean(valid_points, axis=0)

    return center_of_mass
  
import message_filters


class RosInterface:
  def __init__(self):
    self.fruits_to_show = []
    self.cv_bridge = CvBridge()
    self.is_viewpoint_valid = False
    self.fruit_detector = Fruit_detector()
    self.static_fruit_counter = StaticFruitCounter()

    self.marker_array_pub = \
      rospy.Publisher('/icuas2024_plant_fruits', MarkerArray, queue_size=1)
    self.is_world_published = False
    # visualize world
    self.visualize_world_pub = \
      rospy.Publisher('/icuas2024_visualize_world', Marker, queue_size=1)
    # visualize sensor fov
    self.visualize_sensor_fov_pub = \
      rospy.Publisher('/icuas2024_visualize_sensor_fov', Marker, queue_size=1)
    

    self.rgb_sub = message_filters.Subscriber('/red/camera/color/image_raw', Image)
    self.depth_sub = message_filters.Subscriber('/red/camera/depth/image_raw', Image)
    self.odom_sub = message_filters.Subscriber('/red/odometry', Odometry)

    self.ts = message_filters.ApproximateTimeSynchronizer(
      [self.rgb_sub, self.depth_sub, self.odom_sub], 10, 0.05, allow_headerless=True)
    self.ts.registerCallback(self.image_callback)
    
    self.viewpoint_validity_sub = rospy.Subscriber('/icuas2024_viewpoint_validity', Bool, self.validity_callback)
    self.invalid_detection_checker = InvalidDetectionChecker()

  def validity_callback(self, msg):
    self.is_viewpoint_valid = msg.data

  def image_callback(self, rgb_msg, depth_msg, odom):
    self.publish_shelf_in_rviz()  # TODO: move this somewhere else
    self.publish_sensor_fov(odom) # TODO: move this somewhere else
    if not self.is_viewpoint_valid:
      return
    
    # images
    rgb_image = self.cv_bridge.imgmsg_to_cv2(rgb_msg, "bgr8").copy()
    depth_image = self.cv_bridge.imgmsg_to_cv2(depth_msg, "passthrough").copy()
    
    # depth    
    depth_image[depth_image==np.nan] = 0
    np.clip(depth_image, 0, 10, depth_image)
    
    # detect fruits
    fruits, mask_debug, bboxes_drawn, bboxes_drawn2 = \
      self.fruit_detector.detect(rgb_image, depth_image, odom, ['pepper'])
    
    # self.fruit_name = fruit_name
    # self.n_fruit = n_fruit
    # self.detected_time = odom.header.stamp
    # self.xyz = np.array(xyz)
    # self.xyzs = [self.xyz]
    if len(fruits) > 0:
      #print([(fruit.n_fruit, fruit.xyz, fruit.detected_time.to_sec(), fruit.drone_position) for fruit in fruits],",")
      with open("/root/sim_ws/src/icuas24_competition/scripts/fruits_data.txt", "a") as file:
                for fruit in fruits:
                    file.write(f"[{fruit.n_fruit}, {fruit.xyz}, {fruit.detected_time.to_sec()}, {fruit.drone_position}]\n")
                file.write(f"X")



    # static object tracking
    for f in fruits:
      self.static_fruit_counter.add_fruit(f)
    fruits_clustered, centroids, n_fruits = self.static_fruit_counter.cluster_fruits()
    
    # publish fruits
    self.publish_plant_infos([fruit.xyz for fruit in fruits], [fruit.n_fruit for fruit in fruits])

    # mask
    mask_debug = cv2.cvtColor(mask_debug, cv2.COLOR_GRAY2BGR)
    mask_debug = cv2.resize(mask_debug, (640, 480)) # nouse but explicit resizing for future bug

    image_to_show = np.zeros((480, 640, 3), np.uint8)
    #image_to_show = np.zeros((480*2, 640*2, 3), np.uint8)
    # image_to_show[:480, :640] = rgb_image
    # image_to_show[:480, 640:] = bboxes_drawn2
    # image_to_show[480:, :640] = mask_debug
    image_to_show[:480, :640] = bboxes_drawn

    cv2.imshow("image_to_show", image_to_show)
    cv2.waitKey(1)
   

  def publish_plant_infos(self, centroids, n_fruits):
    marker_array = MarkerArray()
    marker_id = 0
    for centroid, n_fruit in zip(centroids, n_fruits):
      marker = Marker()
      marker.header.frame_id = "world"
      marker.header.stamp = rospy.Time.now()
      marker.ns = "fruit"
      marker.id = marker_id
      marker.type = Marker.SPHERE
      marker.action = Marker.MODIFY
      marker.pose.orientation.w = 1.0
      marker.pose.position.x = centroid[0]
      marker.pose.position.y = centroid[1]
      marker.pose.position.z = centroid[2]
      marker.scale.x = self.invalid_detection_checker.fruit_DA_dist_thresh
      marker.scale.y = self.invalid_detection_checker.fruit_DA_dist_thresh
      marker.scale.z = self.invalid_detection_checker.fruit_DA_dist_thresh
      marker.color.a = 1.0
      marker.color.r = 1.0
      marker.color.g = 1.0
      marker.color.b = 0.0
      marker_array.markers.append(marker)
      marker_id += 1
    
      marker = Marker()
      marker.header.frame_id = "world"
      marker.header.stamp = rospy.Time.now()
      marker.ns = "fruit"
      marker.id = marker_id
      marker.type = Marker.TEXT_VIEW_FACING
      marker.text = str(n_fruit)
      marker.action = Marker.MODIFY
      marker.pose.orientation.w = 1.0
      marker.pose.position.x = centroid[0]
      marker.pose.position.y = centroid[1]
      marker.pose.position.z = centroid[2] + 1
      # marker.scale.x = self.invalid_detection_checker.fruit_cluster_dist_thresh
      # marker.scale.y = self.invalid_detection_checker.fruit_cluster_dist_thresh
      marker.scale.z = self.invalid_detection_checker.fruit_cluster_dist_thresh
      # marker.scale.z = 2
      marker.color.a = 1.0
      marker.color.r = 0.0
      marker.color.g = 0.0
      marker.color.b = 0.0
      marker_array.markers.append(marker)
      marker_id += 1
      
    self.marker_array_pub.publish(marker_array)
  
  def publish_shelf_in_rviz(self):
    marker = Marker()
    marker.header.frame_id = "world"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "shelf"
    marker.id = 0
    marker.type = Marker.MESH_RESOURCE
    marker.action = Marker.MODIFY
    marker.lifetime = rospy.Duration(10000)
    marker.pose.orientation.w = 1.0
    marker.pose.position.x = 0
    marker.pose.position.y = 0
    marker.pose.position.z = 0
    marker.scale.x = 1
    marker.scale.y = 1
    marker.scale.z = 1
    marker.color.a = 0.7
    marker.color.r = 1.0
    marker.color.g = 1.0
    marker.color.b = 1.0
    # Warn : resource path is dependent on where you run rviz
    marker.mesh_resource = "package://icuas24_competition/models/icuas24/icuas24_wrld_notxt.dae"
    
    # marker.mesh_use_embedded_materials = True
    self.visualize_world_pub.publish(marker)
  
  def publish_sensor_fov(self, odom):
    marker = Marker()
    marker.header.frame_id = "red/camera" # if you use tf
    # marker.header.frame_id = "world" # if you use odom
    marker.header.stamp = rospy.Time.now()
    marker.ns = "sensor_fov"
    marker.id = 0
    marker.type = Marker.MESH_RESOURCE
    marker.action = Marker.MODIFY
    marker.lifetime = rospy.Duration(10000)
    marker.pose.orientation.w = 1.0
    marker.scale.x = 20
    marker.scale.y = 20
    marker.scale.z = 20
    marker.color.a = 0.4
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    # Warn : resource path is dependent on where you run rviz
    marker.mesh_resource = "package://icuas24_competition/models/camera_fov.stl"
    self.visualize_sensor_fov_pub.publish(marker)

    

if __name__ == '__main__':
  with open("/root/sim_ws/src/icuas24_competition/scripts/fruits_data.txt", "w") as file:
    pass
  rospy.init_node('fruit_detector')
  detector = Fruit_detector()
  
  bridge = CvBridge()

  r = RosInterface()

  rospy.spin()
