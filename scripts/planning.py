import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import rospy
from std_msgs.msg import String
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from sensor_msgs.msg import Image
import tf
import cv2

from sim_ws.src.icuas24_competition.scripts.detector_main import Fruit_detector
import message_filters
from visualization_msgs.msg import MarkerArray, Marker

class ViewpointsArrivalChecker:
  def __init__(self, target_beds_id):
    # bed_id_to_viewpoints
    self.idx_to_x = [1.5, 6.5, 12.5, 18]
    self.idy_to_y = [6  , 14, 21]
    self.idz_to_z = [1.5, 4.0, 6.5]

    self.target_viewpoints = []
    for target_bed_id in target_beds_id:
      viewpoints = self.bed_id_to_viewpoints(target_bed_id)
      self.target_viewpoints.append(viewpoints)

    self.current_target_bed_idx = 0
    self.current_viewpoint_idx = 0

    self.no_more_viewpoint = False
    self.at_beginning = True
    self.goal_changed = False

    ## condition params for altering viewpoint to next one
    self.check_if_time_elapsed_interval = 5.0 # sec
    self.start_time = rospy.Time.now()
    
    self.check_if_goal_reached_dist_threshold = 0.5 # meter
    self.last_odom = None
    self.check_if_goal_reached_yaw_threshold = 20 * np.pi / 180 # rad

  def check_if_time_elapsed(self, start=False):
    if_elapsed = rospy.Time.now() - self.start_time > rospy.Duration(self.check_if_time_elapsed_interval)
    if not start and if_elapsed:
      return True
    if start:
      self.start_time = rospy.Time.now()
    return False
    
  def check_if_goal_reached(self, odom):
    if self.last_odom is None:
      self.last_odom = odom
      return False

    current_target = self.get_current_viewpoint()

    dist = np.linalg.norm(
      np.array([odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z]) - 
      np.array([current_target[0], current_target[1], current_target[2]]
    ))
    
    # get yaw of drone
    q = odom.pose.pose.orientation
    yaw_rad = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
    # get yaw of target
    target_yaw_rad = current_target[3]
    # get yaw difference
    yaw_diff = np.abs(yaw_rad - target_yaw_rad)
    yaw_diff = np.minimum(yaw_diff, 2*np.pi - yaw_diff)
    
    if dist > self.check_if_goal_reached_dist_threshold:
      return False
    if yaw_diff > self.check_if_goal_reached_yaw_threshold:
      return False
    return True
  
  def get_next_viewpoint(self, method, condition_params):
    # return [x, y, z, yaw_rad] at change
    # if last viewpoint is reached, return None
    if self.at_beginning:
      self.at_beginning = False
      return self.get_current_viewpoint() 
  
    if self.no_more_viewpoint:
      return None
    
    if method == "time":
      if self.check_if_time_elapsed():
        self.update_to_next_viewpoint()
    elif method == "distance":
      if self.check_if_goal_reached(condition_params['odom']):
        self.update_to_next_viewpoint()
    
    if self.goal_changed: 
      self.goal_changed = False
      return self.get_current_viewpoint()
    else:
      return None

  def update_to_next_viewpoint(self):
    if self.goal_changed: # drone may stay at goal point
      return
    self.goal_changed = True
    self.current_viewpoint_idx += 1 # switch to next viewpoint
    if self.current_viewpoint_idx >= len(self.target_viewpoints[self.current_target_bed_idx]): # if no more viewpoint in current bed,
      self.current_target_bed_idx += 1 # switch to next bed
      self.current_viewpoint_idx = 0 # reset viewpoint index
      if self.current_target_bed_idx >= len(self.target_viewpoints):
        self.goal_changed = False
        self.no_more_viewpoint = True
        print("final goal reached")
      
  def get_current_viewpoint(self):
    return self.target_viewpoints[self.current_target_bed_idx][self.current_viewpoint_idx]

  def is_valid_viewpoint(self):
    return self.current_viewpoint_idx == 2 or self.current_viewpoint_idx == 6
  
  def bed_id_to_viewpoints(self, index):
    bed_id_ = index - 1
    id_xy = (bed_id_ // 3)
    id_z = (bed_id_ % 3)

    id_x = id_xy // 3
    id_y = id_xy % 3

    viewpoints = [ # x, y, z, yaw_rad
      [self.idx_to_x[id_x]  , self.idy_to_y[id_y]    , 12                      , 0],
      [self.idx_to_x[id_x]  , self.idy_to_y[id_y]-1.5, self.idz_to_z[id_z]     , 0],
      [self.idx_to_x[id_x]  , self.idy_to_y[id_y]+1.5, self.idz_to_z[id_z]     , 0],
      [self.idx_to_x[id_x]  , self.idy_to_y[id_y]    , 12                      , 0],
      [self.idx_to_x[id_x+1], self.idy_to_y[id_y]    , 12                      , np.pi],
      [self.idx_to_x[id_x+1], self.idy_to_y[id_y]-1.5, self.idz_to_z[id_z]     , np.pi],
      [self.idx_to_x[id_x+1], self.idy_to_y[id_y]+1.5, self.idz_to_z[id_z]     , np.pi],
      [self.idx_to_x[id_x+1], self.idy_to_y[id_y]    , 12                      , np.pi],
    ]
    return viewpoints

class view_point_planner:
  def __init__(self):
    self.viewpoint_arrival_checker = None
    self.target_pos_pub         = rospy.Publisher('/red/tracker/input_pose', PoseStamped, queue_size=10)
    self.viewpoint_validity_pub = rospy.Publisher('/icuas2024_viewpoint_validity', Bool, queue_size=10)

    self.sub = rospy.Subscriber('/red/plants_beds', String, self.work_list_callback)

    self.rgb_sub = message_filters.Subscriber('/red/camera/color/image_raw', Image)
    self.depth_sub = message_filters.Subscriber('/red/camera/depth/image_raw', Image)
    self.detection_pub = rospy.Publisher('/red/detection', MarkerArray, queue_size=10)
    self.ts = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], 10, 0.1, allow_headerless=True)
    self.ts.registerCallback(self.image_callback)
    # (subs), queue_size, slop, allow_headerless. slop indicates max allowed delay between messages

    self.odom_sub = rospy.Subscriber('/red/odometry', Odometry, self.odom_callback)
    
    self.cv_bridge = CvBridge()
    self.fruit_detector = Fruit_detector()

  def work_list_callback(self, msg):
    jobs = msg.data.split(' ')
    self.target_fruit = jobs[0].lower()
    
    target_beds = [int(x) for x in jobs[1:]]
    self.viewpoint_arrival_checker = ViewpointsArrivalChecker(target_beds)
    self.publish_view_point(self.viewpoint_arrival_checker.get_current_viewpoint())

  def odom_callback(self, msg):
    if self.viewpoint_arrival_checker is None:
      print('need to receive beds ids')
      return
    
    # self.viewpoint_arrival_checker.get_next_viewpoint("time", None) # this is also possible
    next_viewpoint = self.viewpoint_arrival_checker.get_next_viewpoint("distance", {'odom': msg})
    if next_viewpoint is not None:
      print('next_viewpoint, valid viewpoint: ', next_viewpoint, self.viewpoint_arrival_checker.is_valid_viewpoint())
      self.publish_view_point(next_viewpoint) 
      self.viewpoint_validity_pub.publish(Bool(self.viewpoint_arrival_checker.is_valid_viewpoint())) # validity
    
  # def publish_marker(self, detection_dict):
  #   marker_array = MarkerArray()
  #   for class_name in self.class_list:
  #     center_of_mass_list = detection_dict[class_name]
  #     if len(center_of_mass_list) == 0:
  #       continue
  #     marker = Marker()
  #     marker.header.frame_id = "red/camera"
  #     marker.header.stamp = rospy.Time.now()
  #     marker.ns = class_name
  #     marker.id = 0
  #     marker.type = Marker.SPHERE_LIST
  #     marker.action = Marker.ADD
  #     marker.pose.orientation.w = 1.0
  #     marker.scale.x = 0.1
  #     marker.scale.y = 0.1
  #     marker.scale.z = 0.1
  #     marker.color.a = 1.0
  #     if class_name == "tomato":
  #       marker.color.r = 1.0
  #       marker.color.g = 0.0
  #       marker.color.b = 0.0
  #     elif class_name == "pepper":
  #       marker.color.r = 0.0
  #       marker.color.g = 1.0
  #       marker.color.b = 0.0
  #     elif class_name == "eggplant":
  #       marker.color.r = 0.0
  #       marker.color.g = 0.0
  #       marker.color.b = 1.0
  #     else:
  #       marker.color.r = 0.0
  #       marker.color.g = 0.0
  #       marker.color.b = 0.0
      
    # for center_of_mass in center_of_mass_list:
    #   x, y, z = center_of_mass
    #   marker.points.append(Point(x, y, z))
    #   marker_array.markers.append(marker)
    # self.detection_pub.publish(marker_array)
            
  def image_callback(self, rgb_msg, depth_msg):
    # rgb_image = self.cv_bridge.imgmsg_to_cv2(rgb_msg, "bgr8").copy()
    # depth_image = self.cv_bridge.imgmsg_to_cv2(depth_msg, "passthrough").copy()
    # image_debug = \
    #   self.fruit_detector.detect(rgb_image, depth_image, [self.target_fruit.lower()])
    
    # depth_image[depth_image==np.nan] = 0
    # np.clip(depth_image, 0, 10, depth_image)
    # depth_image_255 = (depth_image/10.0*255).astype(np.uint8)

    
    # # concat rgb and depth
    # depth_image_255 = cv2.cvtColor(depth_image_255, cv2.COLOR_GRAY2BGR)
    # depth_image_255 = cv2.resize(depth_image_255, (640, 480))
    # image_debug = cv2.cvtColor(image_debug, cv2.COLOR_GRAY2BGR)
    # image_debug = cv2.resize(image_debug, (640, 480)) # nouse but explicit resizing for future bug
    # concat_image = np.concatenate((image_debug, depth_image_255), axis=1)
    # cv2.imshow("image_combined", concat_image)
    # cv2.waitKey(1)

    # bbox_dict, detection_dict = self.fruit_detector.detect(rgb_image, depth_image, ['tomato', 'pepper', 'eggplant'])

    # for class_name in self.class_list:
    #   bbox_list = bbox_dict[class_name]
    #   for bbox in bbox_list:
    #     x, y, w, h = bbox
    #     cv2.rectangle(depth_image_255, (x, y), (x+w, y+h), (0, 0, 255), 2)
    #     cv2.putText(depth_image_255, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    #   detection_list = detection_dict[class_name]
    #   print(class_name, detection_list)
    # cv2.imshow("rgb_image", depth_image_255)
    # cv2.waitKey(1)
    # self.publish_marker(detection_dict)
    pass
  
  def publish_view_point(self, x_y_z_yaw):
    pose = PoseStamped()
    pose.header.frame_id = "world"
    pose.header.stamp = rospy.Time.now()
    pose.pose.position.x = x_y_z_yaw[0]
    pose.pose.position.y = x_y_z_yaw[1]
    pose.pose.position.z = x_y_z_yaw[2]

    quat = tf.transformations.quaternion_from_euler(0, 0, x_y_z_yaw[3])
    pose.pose.orientation.x = quat[0]
    pose.pose.orientation.y = quat[1]
    pose.pose.orientation.z = quat[2]
    pose.pose.orientation.w = quat[3]

    self.target_pos_pub.publish(pose)



if __name__ == '__main__':
    rospy.init_node('simple_view_point_publisher')
    view_point_planner = view_point_planner()
    rospy.spin()
