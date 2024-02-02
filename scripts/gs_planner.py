import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Point, Transform
from nav_msgs.msg import Odometry
import re
import time
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from sensor_msgs.msg import Image
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint

import message_filters
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Bool
import math
from std_msgs.msg import Int32MultiArray




class view_point_planner:
    def __init__(self):
        
        self.odom_sub = rospy.Subscriber('/red/odometry', Odometry, self.odom_callback)
        self.sub = rospy.Subscriber('/red/plants_beds', String, self.work_list_callback)
        self.pub = rospy.Publisher('/red/tracker/input_pose', PoseStamped, queue_size=10)
        self.traj_pub = rospy.Publisher("/red/tracker/input_trajectory", MultiDOFJointTrajectory, queue_size=10)

        self.cv_bridge = CvBridge()

        self.waypoints = [] 
        self.waypoints_to_show = [] 
        self.waypoints_raw = []
        self.current_waypoint = 0
        self.current_viewpoint = 0

        self.visualize_waypoint_array = rospy.Publisher("/waypoint_array", MarkerArray, queue_size=10)
        self.viewpoint_validity_pub = rospy.Publisher('/icuas2024_target_viewpoint', Int32MultiArray, queue_size=10)

        self.is_viewpoint = False

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
        self.position_thereshold = 1 # meter
        self.orientation_threshold= 0.3 # 0.2 rad = 11.5 degree
        self.first_pub_flag = True
        self.is_start = False
    

    def quaternion_to_euler(self, q):
        x,y,z,w = q
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(t0, t1) *180/np.pi
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = np.arcsin(t2) *180/np.pi
        
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(t3, t4) *180/np.pi
        
        return [roll, pitch, yaw]
    
    
    def quaternion_multiply(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return [w, x, y, z]

    def quaternion_conjugate(self, q):
        w, x, y, z = q
        return [w, -x, -y, -z]


    def calc_pose_error(self, pose1, pose2):
        position_error = np.linalg.norm(pose1[0:3] - pose2[0:3])
        # calc quaternion error
        q1 = pose1[3:7]
        q2 = pose2[3:7]
        q_diff = self.quaternion_multiply(q2, self.quaternion_conjugate(q1))
        r_error, p_error, y_error = self.quaternion_to_euler(q_diff)
        r_error, p_error, y_error = map(abs, [r_error, p_error, y_error])
        if r_error > 90:
            r_error = 180 - r_error
        # print('position_1', pose1[0:3])
        # print('position_2', pose2[0:3])
        # print("position error: ", position_error, " orientation error: ", theta_error)
        
        if position_error < self.position_thereshold and y_error < self.orientation_threshold and p_error < self.orientation_threshold and r_error < self.orientation_threshold:
            #print(r_error, p_error, y_error)
            return True
        else:
            return False
        


    
    def odom_callback(self, msg):
        self.pose = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])

    
    def publish_way_point(self, xyz, direction = 1, is_viewpoint=2):
        pose = PoseStamped()
        pose.header.frame_id = "world"
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = xyz[0]
        pose.pose.position.y = xyz[1]
        pose.pose.position.z = xyz[2]
        if direction == 1:
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = 0.0
            pose.pose.orientation.w = 1.0
        else:
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = 1.0
            pose.pose.orientation.w = 0.0
        self.pub.publish(pose)
        print("publishing view point axis: ", direction, " xyz: ", xyz)
        time.sleep(0.06)
        self.pub.publish(pose)
        self.first_pub_flag = False

    def waypoints_to_traj_msg(self, waypoints):
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



    def view_point_vaildation(self, xyz, direction = 1, is_viewpoint=2):
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

        if is_viewpoint == 2:
            self.position_thereshold = 1.0
            self.orientation_threshold = 10
        elif is_viewpoint == 1:
            self.position_thereshold = 3
            self.orientation_threshold = 45
        else:
            self.position_thereshold = 3
            self.orientation_threshold = 45
        
        return self.calc_pose_error(self.pose, np.array([pose.pose.position.x, 
                                                         pose.pose.position.y, 
                                                         pose.pose.position.z, 
                                                         pose.pose.orientation.x, 
                                                         pose.pose.orientation.y, 
                                                         pose.pose.orientation.z, 
                                                         pose.pose.orientation.w]))
    
    def waypoint_vaildation(self, xyz, direction = 1, is_viewpoint=2):
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

        if is_viewpoint == 2:
            self.position_thereshold = 0.8
            self.orientation_threshold = 5
        elif is_viewpoint == 1:
            self.position_thereshold = 3
            self.orientation_threshold = 45
        else:
            self.position_thereshold = 3.5
            self.orientation_threshold = 45
        
        return self.calc_pose_error(self.pose, np.array([pose.pose.position.x, 
                                                         pose.pose.position.y, 
                                                         pose.pose.position.z, 
                                                         pose.pose.orientation.x, 
                                                         pose.pose.orientation.y, 
                                                         pose.pose.orientation.z, 
                                                         pose.pose.orientation.w]))


    def check_and_move_to_start(self):
        start_x, start_y, start_z = 0, 0,  2
        position_tolerance = 2

        time.sleep(1) 
        current_x, current_y, current_z = self.pose[0], self.pose[1], self.pose[2]
        print(current_x, current_y)


        if abs(current_x - start_x) > position_tolerance or \
           abs(current_y - start_y) > position_tolerance or \
           abs(current_z - start_z) > position_tolerance:

            ascend_z = start_z + 8
            self.publish_way_point([current_x, current_y, ascend_z])
            rospy.sleep(5) 

            self.publish_way_point([start_x, start_y, ascend_z])
            rospy.sleep(5)  


            self.publish_way_point([start_x, start_y, start_z])
            rospy.sleep(10)  
        print("start point chk")


    def wait_for_detection(self, t):
        rospy.sleep(t)
    

    def work_list_callback(self, msg):
        rospy.sleep(0.1)
        work_str = msg.data
        work_list = re.findall(r'\d+', msg.data)
        work_list = [int(num) for num in work_list]
        work_list_opp = [x+9 for x in work_list]
        work_list_with_1 = [(x,1) for x in work_list]
        work_list_with_opp = [(x,-1) for x in work_list_opp]
        combined_list = work_list_with_1 + work_list_with_opp
        sorted_combined_list = sorted(combined_list, key=lambda x: x[0])
        
        for work_index in sorted_combined_list:
            index = work_index[0]
            direction = work_index[1]
            xi, yi, zi = self.get_xiyizi_from_index(index)
            if self.current_xi != xi: 
                # need to move to virtual point
                new_goal = [self.current_xi*self.size_x+self.offset_x, 0, zi*self.size_z +self.offset_z]
                self.waypoints_raw.append(new_goal)
                self.waypoints.append([new_goal, direction, 0])

                self.current_xi = xi
                self.cuurent_yi = 0
                self.current_zi = zi

                new_goal = [xi*self.size_x + self.offset_x, 0, zi*self.size_z + self.offset_z]
                self.waypoints_raw.append(new_goal)
                new_goal[0] += 1.8
                new_goal[1] += 1
                self.waypoints.append([new_goal, direction, 1])

                new_goal = self.get_position_from_xiyizi([xi, yi, zi])
                self.waypoints_raw.append(new_goal)
                new_goal[0] -= 0.6 if direction >= 0.5 else -0.6
                new_goal[2] +=0.3
                self.waypoints.append([new_goal, direction, 2])
                self.current_yi = yi


            else:
                new_goal = self.get_position_from_xiyizi([xi, yi, zi])
                self.waypoints_raw.append(new_goal)
                new_goal[0] -= 0.6 if direction >= 0.5 else -0.6
                new_goal[2] +=0.3
                self.waypoints.append([new_goal, direction, 2])
        
        new_goal = [self.current_xi*self.size_x + self.offset_x, 0, (self.current_zi*self.size_z + self.offset_z*2)/2]
        self.waypoints_raw.append(new_goal)
        self.waypoints.append([new_goal, direction, 0])
        
        new_goal = [self.offset_x, 0, self.offset_z]
        self.waypoints_raw.append(new_goal)
        self.waypoints.append([new_goal, 1, 0])
        self.check_and_move_to_start()
        print(self.waypoints_raw)
        print("start")
        now = rospy.Time.now().to_sec()
        rate = rospy.Rate(60) 
        start_view = 0
        self.is_viewpoint = 0
        # self.traj_pub.publish(self.waypoints_to_traj_msg(self.waypoints))
        # rospy.sleep(0.1)
        # self.traj_pub.publish(self.waypoints_to_traj_msg(self.waypoints))
        msg = Int32MultiArray()
        msg.data = [111, int(self.is_viewpoint)]
        self.viewpoint_validity_pub.publish(msg)
        self.publish_way_point(self.waypoints[self.current_waypoint][0], self.waypoints[self.current_waypoint][1], self.waypoints[self.current_waypoint][2])
        while not rospy.is_shutdown():
            if self.waypoint_vaildation(self.waypoints[self.current_waypoint][0], self.waypoints[self.current_waypoint][1], self.waypoints[self.current_waypoint][2]):
                if self.is_viewpoint:
                    if rospy.Time.now().to_sec() - start_view < 0.5:
                        continue
                self.current_viewpoint = self.current_waypoint
                self.current_viewpoint += 1
                self.is_viewpoint = 0
                self.current_waypoint += 1 
                if self.current_waypoint >= len(self.waypoints):
                    break
                #print("time:" + str(rospy.Time.now().to_sec() -now))
                
                self.publish_way_point(self.waypoints[self.current_waypoint][0], self.waypoints[self.current_waypoint][1], self.waypoints[self.current_waypoint][2])
                self.publish_waypoint_in_rviz(self.waypoints[self.current_waypoint][0])

            if self.view_point_vaildation(self.waypoints[self.current_viewpoint][0], self.waypoints[self.current_viewpoint][1], self.waypoints[self.current_viewpoint][2]):
                if self.waypoints[self.current_viewpoint][2]==2:
                    if not self.is_viewpoint:
                        start_view = rospy.Time.now().to_sec()
                    self.is_viewpoint = 1
                    self.viewpoint_validity_pub.publish(msg)
            else:
                self.is_viewpoint = 0
                self.viewpoint_validity_pub.publish(msg)
            
            rate.sleep()

        self.viewpoint_validity_pub.publish(Int32MultiArray[999, int(self.is_viewpoint)])
        print("time:" + str(rospy.Time.now().to_sec() -now))
        with open("/root/sim_ws/src/icuas24_competition/scripts/OUT.txt", "a") as file:
            file.write(f"{str(rospy.Time.now().to_sec() -now)},")
        rospy.signal_shutdown("end!")
        exit()


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
        
    def publish_waypoint_in_rviz(self, point):

        delete_markers = MarkerArray()
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        delete_markers.markers.append(delete_marker)
        self.visualize_waypoint_array.publish(delete_markers)
        rospy.sleep(0.01)
        
        self.waypoints_to_show.append(point)

        marker_array = MarkerArray()

        for i, point in enumerate(self.waypoints_to_show):
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
            marker.color.a = max(i-(len(self.waypoints_to_show)-10),0)/10
            marker.color.r = 1.0
            marker.color.g = 0
            marker.color.b = 0
            
            marker_array.markers.append(marker)

        self.visualize_waypoint_array.publish(marker_array)

if __name__ == '__main__':
    rospy.init_node('simple_view_point_publisher')
    planner = view_point_planner()
    rospy.spin()