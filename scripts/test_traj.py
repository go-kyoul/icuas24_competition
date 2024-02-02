import rospy
import numpy as np
import time
from geometry_msgs.msg import PoseStamped, Twist, Transform
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from std_msgs.msg import String

rospy.init_node("test_traj")
# my_pub = rospy.Publisher("/red/tracker/input_pose", PoseStamped, queue_size=10)

# my_msg = PoseStamped()
# my_msg.pose.position.x = 0
# my_msg.pose.position.y = 0
# my_msg.pose.position.z = 4.0
# my_msg.pose.orientation.x = 0
# my_msg.pose.orientation.y = 0
# my_msg.pose.orientation.z = 0
# my_msg.pose.orientation.w = 1




waypoints = [[[13,6,20],1],
             [[0,0,20],1],]

my_pub = rospy.Publisher("/red/tracker/input_trajectory", MultiDOFJointTrajectory, queue_size=10)

def waypoints_to_traj_msg(waypoints):
    my_msg = MultiDOFJointTrajectory()
    points = []
    for i, waypoint in enumerate(waypoints):
        point = MultiDOFJointTrajectoryPoint()
        transform = Transform()
        transform.translation.x = waypoint[0][0]
        transform.translation.y = waypoint[0][1]
        transform.translation.z = waypoint[0][2]
        if waypoints[1] == 1:
            direction = [0,0,0,1]
        else:
            direction = [0,0,1,0]
        transform.rotation.x = direction[0]
        transform.rotation.y = direction[1]
        transform.rotation.z = direction[2]
        transform.rotation.w = direction[3]
        point.transforms = [transform]
        
        twist = Twist()
        twist.linear.x = 0
        twist.linear.y = 0
        twist.linear.z = 0
        point.velocities = [twist]


        point.time_from_start = rospy.Duration(50.0*i)

        points.append(point)

    my_msg.points = points
    
    return my_msg

output = waypoints_to_traj_msg(waypoints)

my_pub.publish(output)
time.sleep(0.1)
my_pub.publish(output)
