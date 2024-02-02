import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Odometry
import re
import time
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from sensor_msgs.msg import Image

from sim_ws.src.icuas24_competition.scripts.detector_main import Fruit_detector
import message_filters
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import PoseStamped, Twist, Transform
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from std_msgs.msg import String
now = time.time()
def waypoints_to_traj_msg(waypoints):
    my_msg = MultiDOFJointTrajectory()
    points = []
    for waypoint in waypoints:
        point = MultiDOFJointTrajectoryPoint()
        transform = Transform()
        transform.translation.x = waypoint[0][0]
        transform.translation.y = waypoint[0][1]
        transform.translation.z = waypoint[0][2]
        # print(waypoint[1])
        if waypoint[1] == 1:
            direction = [0,0,0,1]
            # print('front')
        else:
            direction = [0,0,1,0]
            # print('back')
        transform.rotation.x = direction[0]
        transform.rotation.y = direction[1]
        transform.rotation.z = direction[2]
        transform.rotation.w = direction[3]
        point.transforms = [transform]
        points.append(point)
    my_msg.points = points
    
    return my_msg

class view_point_planner:
    def __init__(self):
        self.odom_sub = rospy.Subscriber('/red/odometry', Odometry, self.odom_callback)
        self.sub = rospy.Subscriber('/red/plants_beds', String, self.work_list_callback)
        self.pub = rospy.Publisher('/red/tracker/input_pose', PoseStamped, queue_size=10)
        self.traj_pub = rospy.Publisher("/red/tracker/input_trajectory", MultiDOFJointTrajectory, queue_size=10)
        self.cv_bridge = CvBridge()

        self.waypoints = [] 
        self.visualize_waypoint_array = rospy.Publisher("/waypoint_array", MarkerArray, queue_size=10)

        self.publish_interval = rospy.Duration(30.0)
        self.last_publish_time = rospy.Time.now()
        self.current_xi = 0
        self.current_yi = 0
        self.current_zi = 0
        self.size_x = 6
        self.size_y = 7.4
        self.size_z = 2.6
        self.offset_x = 1.0
        self.offset_y = 6.0
        self.offset_z = 1.8

        self.pose = np.array([0, 0, 0, 0, 0, 0, 1])

        self.is_drone_reached = False
        self.position_thereshold = 0.5 # meter
        self.orientation_threshold = 0.2 # 0.2 rad = 11.5 degree
        self.first_pub_flag = True
    
    def calc_pose_error(self, pose1, pose2):
        position_error = np.linalg.norm(pose1[0:3] - pose2[0:3])
        # calc quaternion error
        q1 = pose1[3:7]
        q2 = pose2[3:7]
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        theta_error = np.arccos(2*np.dot(q1, q2)**2 - 1)
        print('position_1', pose1[0:3])
        print('position_2', pose2[0:3])
        print("position error: ", position_error, " orientation error: ", theta_error)
        if position_error < self.position_thereshold and theta_error < self.orientation_threshold:
            return True
        else:
            return False
        
    def publish_waypoint_in_rviz(self, position):

        delete_markers = MarkerArray()
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        delete_markers.markers.append(delete_marker)
        self.visualize_waypoint_array.publish(delete_markers)
        rospy.sleep(0.05)
        self.waypoints.append(position)

        marker_array = MarkerArray()

        for i, point in enumerate(self.waypoints):
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "waypoint"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.MODIFY
            marker.lifetime = rospy.Duration(10000)
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = point[0]
            marker.pose.position.y = point[1]
            marker.pose.position.z = point[2]
            marker.scale.x = 0.4
            marker.scale.y = 0.4
            marker.scale.z = 0.4
            marker.color.a = 1
            marker.color.r = 1.0
            marker.color.g = 0
            marker.color.b = 0
            
            marker_array.markers.append(marker)
            

        self.visualize_waypoint_array.publish(marker_array)


    
    def odom_callback(self, msg):
        self.pose = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])

    def view_point_vaildation(self, xyz, direction = 1):
        pose = PoseStamped()
        pose.header.frame_id = "world"
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = xyz[0]
        pose.pose.position.y = xyz[1]
        pose.pose.position.z = xyz[2]
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        if direction == 1:
            pose.pose.orientation.z = 0.0
            pose.pose.orientation.w = 1.0
        else:
            pose.pose.orientation.z = 1.0
            pose.pose.orientation.w = 0.0
        
        while not self.calc_pose_error(self.pose, np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z, pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w])):
            time.sleep(0.05)
        print(time.time() - now)

    def publish_view_point(self, xyz, direction = 1):
        pose = PoseStamped()
        pose.header.frame_id = "world"
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = xyz[0]
        pose.pose.position.y = xyz[1]
        pose.pose.position.z = xyz[2]
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        if direction == 1:
            pose.pose.orientation.z = 0.0
            pose.pose.orientation.w = 1.0
        else:
            pose.pose.orientation.z = 1.0
            pose.pose.orientation.w = 0.0
        self.pub.publish(pose)
        print("publishing view point axis: ", direction, " xyz: ", xyz)
        if self.first_pub_flag:
            time.sleep(0.10)
            self.pub.publish(pose)
            self.first_pub_flag = False

        self.view_point_vaildation(xyz, direction)
    

    def work_list_callback(self, msg):
        work_str = msg.data
        work_list = re.findall(r'\d+', msg.data)
        work_list = [int(num) for num in work_list]
        work_list_opp = [x+9 for x in work_list]
        work_list_with_1 = [(x,1) for x in work_list]
        work_list_with_opp = [(x,-1) for x in work_list_opp]
        combined_list = work_list_with_1 + work_list_with_opp
        sorted_combined_list = sorted(combined_list, key=lambda x: x[0])
        print(sorted_combined_list)
        waypoint_list = []
        for work_index in sorted_combined_list:
            index = work_index[0]
            direction = work_index[1]
            xi, yi, zi = self.get_xiyizi_from_index(index)
            if self.current_xi != xi: 
                
                # need to move to virtual point
                new_goal = [self.current_xi*self.size_x+self.offset_x, 0, zi*self.size_z +self.offset_z]
                self.publish_waypoint_in_rviz([new_goal[0]+0.5, new_goal[1]+2, new_goal[2]])
                waypoint_list.append([[new_goal[0]+0.5, new_goal[1]+2, new_goal[2]],direction])

                self.current_xi = xi
                self.cuurent_yi = 0
                self.current_zi = zi

                new_goal = [xi*self.size_x + self.offset_x, 0, zi*self.size_z + self.offset_z]
                self.publish_waypoint_in_rviz([new_goal[0]-0.5, new_goal[1]+2, new_goal[2]])
                # self.publish_view_point(new_goal, 1)
                waypoint_list.append([[new_goal[0]-0.5, new_goal[1]+2, new_goal[2]],direction])

                new_goal = self.get_position_from_xiyizi([xi, yi, zi])
                new_goal[0]-= 0.8 if direction >= 0.5 else -0.8
                new_goal[2]+=0.5
                waypoint_list.append([[new_goal[0], new_goal[1], new_goal[2]],direction])
                self.publish_waypoint_in_rviz(new_goal)
                # new_goal[1]=new_goal[1]+2
                # waypoint_list.append([[new_goal[0], new_goal[1], new_goal[2]],direction])
                # self.publish_waypoint_in_rviz(new_goal)
                self.current_yi = yi

                # self.traj_pub.publish(waypoints_to_traj_msg(waypoint_list))
                # rospy.sleep(0.1)
                # self.traj_pub.publish(waypoints_to_traj_msg(waypoint_list))
                # self.view_point_vaildation(new_goal, direction)


            else:
                new_goal = self.get_position_from_xiyizi([xi, yi, zi])
                new_goal[0]-= 0.8 if direction >= 0.5 else -0.8
                new_goal[2]+=0.5
                waypoint_list.append([[new_goal[0], new_goal[1], new_goal[2]],direction])
                self.publish_waypoint_in_rviz(new_goal)
                # new_goal[1]=new_goal[1]+2
                # waypoint_list.append([[new_goal[0], new_goal[1], new_goal[2]],direction])
                # self.publish_waypoint_in_rviz(new_goal)
                #self.publish_view_point(new_goal, direction)

        #waypoint_list = []
        
        new_goal = [self.current_xi*self.size_x + self.offset_x, 0, self.current_zi*self.size_z + self.offset_z]
        self.publish_waypoint_in_rviz(new_goal)
        waypoint_list.append([[new_goal[0], new_goal[1]+2, new_goal[2]],direction])
        #self.publish_view_point(new_goal, -1)
        new_goal = [self.offset_x, 0, self.offset_z]
        self.publish_waypoint_in_rviz(new_goal)
        waypoint_list.append([[new_goal[0], new_goal[1], new_goal[2]],1])
        
        self.traj_pub.publish(waypoints_to_traj_msg(waypoint_list))
        rospy.sleep(0.1)
        self.traj_pub.publish(waypoints_to_traj_msg(waypoint_list))
        rospy.sleep(5)
        self.view_point_vaildation(new_goal, 1)
        self.publish_view_point(new_goal, 1)
        



    def get_position_from_xiyizi(self, xiyizi):
        x = xiyizi[0]*self.size_x + self.offset_x
        y = xiyizi[1]*self.size_y + self.offset_y
        z = xiyizi[2]*self.size_z + self.offset_z
        return [x, y, z]

    def get_xiyizi_from_index(self, index):
        xi = ((index -1)//9)
        yi = ((index -1)%9//3)
        zi = ((index -1)%3)
        return int(xi), int(yi), int(zi)
        

if __name__ == '__main__':
    rospy.init_node('simple_view_point_publisher')
    view_point_planner = view_point_planner()
    rospy.spin()