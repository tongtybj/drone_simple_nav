import os
import sys
current_path = os.path.abspath(os.path.dirname(__file__))[:-9]  # -9 removes '/ros_node'
sys.path.insert(0, current_path)
from sensor_msgs.msg import PointCloud2
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from planner.msg import *
import actionlib
from nav_msgs.msg import Odometry, Path
from map_server.pcl_server import PCLServer
import time
from pyquaternion import Quaternion
from mavros_msgs.srv import SetMode, SetModeRequest
from mavros_msgs.msg import State, PositionTarget
import numpy as np
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from visualizer.visualizer import Visualizer
from traj_planner.geo_planner import GeoPlanner
from map_server.octree_server import OctreeServer
from octomap_msgs.msg import Octomap
from geometry_msgs.msg import PoseStamped

class DroneState():
    def __init__(self):
        self.global_pos = np.zeros(3)
        self.global_vel = np.zeros(3)
        self.local_vel = np.zeros(3)
        self.attitude = Quaternion()  # ref: http://kieranwynn.github.io/pyquaternion/
        self.yaw = 0.0


class AstarConfig():
    def __init__(self):
        self.resolution = rospy.get_param("~resolution", 0.2)
        self.min_flight_height = rospy.get_param("~min_flight_height", 2.0)
        self.max_flight_height = rospy.get_param("~max_flight_height", 4.0)


class TrajPlanner():
    def __init__(self, node_name="traj_planner"):
        # Node
        rospy.init_node(node_name, anonymous=False)

        # Parameters
        a_star_config = AstarConfig()
        collision_threshold = rospy.get_param("~collision_threshold", 0.4)
        self.plan_mode = rospy.get_param("~plan_mode", 'segment')
        self.longitu_step_dis = rospy.get_param("~longitu_step_dis", 5.0)  # the distance forward in each replanning
        self.lateral_step_length = rospy.get_param("~lateral_step_length", 1.0)  # if local target pos in obstacle, take lateral step
        self.global_target_reach_threshold = rospy.get_param("~global_target_reach_threshold", 0.2)
        self.local_target_reach_threshold = rospy.get_param("~local_target_reach_threshold", 1.0)
        self.cmd_hz = rospy.get_param("~cmd_hz", 60)
        self.yaw_shift_tol = rospy.get_param("~yaw_shift_tol", 0.17453)
        self.move_vel = rospy.get_param("~move_vel", 1.0)
        self.target_pos_z = rospy.get_param("~target_pos_z", 2.0)
        map_server = rospy.get_param("~map_server", "octomap")  # available options: octomap, pcl

        if map_server == "octomap":
            rospy.loginfo("Using Octomap to perform collision check")
            self.map = OctreeServer(collision_threshold)
            self.octomap_sub = rospy.Subscriber('/octomap_binary', Octomap, self.map.octomap_cb)

        elif map_server == "pcl":
            rospy.loginfo("Using PCL to perform collision check")
            self.map = PCLServer(collision_threshold)
            self.pcl_sub = rospy.Subscriber('/pointcloud/output', PointCloud2, self.map.pcl_cb)

        else:
            rospy.logerr("Invalid map_server! Please specify 'octomap' or 'pcl'")

        # Planner
        self.planner = GeoPlanner(a_star_config, self.move_vel, self.cmd_hz)
        self.visualizer = Visualizer()
        self.drone_state = DroneState()
        self.state_cmd = PositionTarget()
        self.state_cmd.coordinate_frame = 1
        self.des_path = Path()
        self.init_marker_arrays()
        self.target_state = None

        # Flags and counters
        self.target_received = False
        self.reached_target = False
        self.near_global_target = False
        self.odom_received = False

        # Server
        self.plan_server = actionlib.SimpleActionServer('plan', PlanAction, self.execute_mission, False)
        self.plan_server.start()

        # Subscribers
        self.flight_state_sub = rospy.Subscriber('mavros/state', State, self.flight_state_cb)
        self.odom_sub = rospy.Subscriber('mavros/local_position/odom', Odometry, self.odom_cb)

        # Publishers
        self.local_pos_cmd_pub = rospy.Publisher("mavros/setpoint_raw/local", PositionTarget, queue_size=10)
        self.pose_cmd_pub = rospy.Publisher("target_pose", PoseStamped, queue_size=1)
        self.target_vis_pub = rospy.Publisher('global_target', Marker, queue_size=10)
        self.local_target_pub = rospy.Publisher('local_target', Marker, queue_size=10)
        self.target_path_pub = rospy.Publisher("target_path", Path, queue_size=1)


        self.raw_path_pub = rospy.Publisher("raw_path", Path, queue_size=1)
        self.prune_path_pub = rospy.Publisher("prune_path", Path, queue_size=1)

        rospy.loginfo(f"Trajectory planner initialized!")

    def flight_state_cb(self, data):
        self.flight_state = data

    def odom_cb(self, data):
        '''
        1. store the drone's global status
        2. publish dynamic tf transform from map frame to camera frame
        (Currently, regard camera frame as drone body frame)
        '''
        self.odom_received = True
        self.odom = data
        local_pos = np.array([data.pose.pose.position.x,
                              data.pose.pose.position.y,
                              data.pose.pose.position.z])
        global_pos = local_pos
        local_vel = np.array([data.twist.twist.linear.x,
                              data.twist.twist.linear.y,
                              data.twist.twist.linear.z])
        quat = Quaternion(data.pose.pose.orientation.w,
                          data.pose.pose.orientation.x,
                          data.pose.pose.orientation.y,
                          data.pose.pose.orientation.z)  # from local to global
        global_vel = quat.rotate(local_vel)
        self.drone_state.global_pos = global_pos
        #rospy.loginfo("global_pos: {}".format(global_pos))
        self.drone_state.global_vel = global_vel
        self.drone_state.local_vel = local_vel
        self.drone_state.attitude = quat

        # get yaw from quaternion
        euler = euler_from_quaternion([data.pose.pose.orientation.x,
                                       data.pose.pose.orientation.y,
                                       data.pose.pose.orientation.z,
                                       data.pose.pose.orientation.w])

        self.drone_state.yaw = euler[2]

        if self.target_received and np.linalg.norm(global_pos - self.global_target) < self.global_target_reach_threshold:
            rospy.loginfo("Global target reached!\n")
            self.end_mission(reached_target=True)

    def init_mission(self):
        self.target_received = True
        self.reached_target = False
        self.near_global_target = False

        current_pos = self.drone_state.global_pos
        self.target_state = np.array([current_pos, [0,0,0]])

    def end_mission(self, reached_target):
        self.target_received = False
        self.reached_target = reached_target
        self.near_global_target = False
        if self.plan_mode == 'segment':
            self.replan_timer.shutdown()

    def execute_mission(self, goal):
        target = goal.target

        self.global_target = np.array([target.pose.position.x, target.pose.position.y, self.target_pos_z])  # 3d pos
        rospy.loginfo("Target received: x = %f, y = %f, z = %f", target.pose.position.x, target.pose.position.y, self.target_pos_z)
        self.vis_target()

        self.init_mission()

        if self.plan_mode == 'global':
            rospy.loginfo("Plan mode: global")
            self.global_planning()
        elif self.plan_mode == 'segment':
            rospy.loginfo("Plan mode: segment")
            self.periodic_planning()
        else:
            rospy.logerr("Invalid replan_mode!")

        self.report_planning_result()

    def vis_target(self):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.pose.position.x = self.global_target[0]
        marker.pose.position.y = self.global_target[1]
        marker.pose.position.z = self.target_pos_z
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.4
        marker.scale.y = 0.4
        marker.scale.z = 0.4
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        self.target_vis_pub.publish(marker)

    def report_planning_result(self):
        while not self.plan_server.is_preempt_requested() and not self.reached_target:
            time.sleep(0.01)

        if self.plan_server.is_preempt_requested():
            rospy.loginfo("Planning preempted!\n")
            self.end_mission(reached_target=False)
            self.plan_server.set_preempted()
        else:  # this means the target is reached
            result = PlanResult()
            result.success = self.reached_target
            self.plan_server.set_succeeded(result)

    def global_planning(self):
        while not self.odom_received:
            time.sleep(0.01)

        self.target_state = np.array([self.global_target, np.zeros(3)])  # [3d pos, 3d vel]

        self.plan()

    def periodic_planning(self):
        while not self.odom_received:
            time.sleep(0.01)

        self.replan_timer = rospy.Timer(rospy.Duration(0.01), self.replan_cb)

    def replan_cb(self, event):

        if self.reached_target or self.near_global_target:
            return

        if self.plan_server.is_preempt_requested():
            return

        seed = 0
        replan_flag = self.set_local_target(seed)

        if not replan_flag: # no need to replan
            return


        while True:
            try:
                self.plan()
                break
            except Exception as ex:
                rospy.logwarn("Local planning failed: %s", ex)
                seed += 1
                self.set_local_target(seed)
                if seed > 10:
                    rospy.logerr("Entire planning failed!\n")
                    self.end_mission(reached_target=False)
                    self.plan_server.set_aborted()
                    return

    def set_local_target(self, seed=0):
        current_pos = self.drone_state.global_pos
        current_pos_2d = current_pos[:2]
        global_target_pos = self.global_target
        global_target_pos_2d = global_target_pos[:2]
        local_target_pos = self.target_state[0]


        # if current pos is far from the local target, do not need to update the local target
        if np.linalg.norm(local_target_pos - current_pos) > self.local_target_reach_threshold:  # 3d distance
            return False

        # if current pos is close enough to global target, set local target as global target
        if np.linalg.norm(global_target_pos - current_pos) < self.longitu_step_dis:  # 3d distance
            self.target_state[0] = global_target_pos
            self.near_global_target = True
            return True


        longitu_dir = (global_target_pos_2d - current_pos_2d)/np.linalg.norm(global_target_pos_2d - current_pos_2d)
        lateral_dir = np.array([[longitu_dir[1], -longitu_dir[0]],
                                [-longitu_dir[1], longitu_dir[0]]])
        lateral_dir_flag = 0
        lateral_move_dis = self.lateral_step_length


        # get local target pos
        if seed > 1e-3:
            local_target_pos_2d = current_pos_2d + self.longitu_step_dis * longitu_dir + np.random.normal(0, 1, 2)  # 0 for mean, 1 for std
        else:
            local_target_pos_2d = current_pos_2d + self.longitu_step_dis * longitu_dir

        # expand local_target_pos_2d to a 3-dimensional array
        local_target_pos = np.append(local_target_pos_2d, self.target_pos_z)

        while self.map.has_collision_strict(local_target_pos):
            local_target_pos_2d += lateral_move_dis * lateral_dir[lateral_dir_flag]
            lateral_dir_flag = 1 - lateral_dir_flag
            lateral_move_dis += self.lateral_step_length
            local_target_pos = np.append(local_target_pos_2d, self.target_pos_z)

        # get local target vel
        goal_dir = (global_target_pos - local_target_pos) / np.linalg.norm(global_target_pos - local_target_pos)
        local_target_vel = self.move_vel * goal_dir

        local_target = np.array([local_target_pos,
                                 local_target_vel])

        self.target_state = local_target

        self.visualize_local_target()

        return True

    def plan(self):

        time_start = time.time()
        raw_path, prune_path = self.planner.geo_traj_plan(self.map, self.drone_state, self.target_state)
        time_end = time.time()
        rospy.loginfo("Planning time: {}".format(time_end - time_start))

        # send the path command 
        target_path_msg = Path()
        target_path_msg.header.frame_id = "world"
        t = rospy.Time.now().to_sec()
        target_path_msg.header.stamp = rospy.Time.from_sec(t)
        for i, pos in enumerate(prune_path[1:]):

            pose = PoseStamped()
            pose.header.seq = i
            pose.header.frame_id = target_path_msg.header.frame_id
            prev_pos = np.array(prune_path[i])
            pos = np.array(pos)
            t += np.linalg.norm(prev_pos - pos) / self.move_vel
            pose.header.stamp = rospy.Time.from_sec(t)

            pose.pose.position.x = pos[0]
            pose.pose.position.y = pos[1]
            pose.pose.position.z = pos[2]
            pose.pose.orientation.w = 1
            target_path_msg.poses.append(pose)

        self.target_path_pub.publish(target_path_msg)


        # visualize path
        raw_path_msg = Path()
        raw_path_msg.header.frame_id = "map"
        raw_path_msg.header.stamp = rospy.Time.now()
        for pos in raw_path:
            pose = PoseStamped()
            pose.header = raw_path_msg.header
            pose.pose.position.x = pos[0]
            pose.pose.position.y = pos[1]
            pose.pose.position.z = pos[2]
            pose.pose.orientation.w = 1
            raw_path_msg.poses.append(pose)
        self.raw_path_pub.publish(raw_path_msg)

        prune_path_msg = Path()
        prune_path_msg.header.frame_id = "map"
        prune_path_msg.header.stamp = rospy.Time.now()
        for pos in prune_path:
            pose = PoseStamped()
            pose.header = prune_path_msg.header
            pose.pose.position.x = pos[0]
            pose.pose.position.y = pos[1]
            pose.pose.position.z = pos[2]
            pose.pose.orientation.w = 1
            prune_path_msg.poses.append(pose)
        self.prune_path_pub.publish(prune_path_msg)

        return raw_path, prune_path

    def init_marker_arrays(self):
        # local target
        self.local_target_marker = Marker()
        self.local_target_marker.header.frame_id = "map"
        self.local_target_marker.type = Marker.SPHERE
        self.local_target_marker.scale.x = 0.4
        self.local_target_marker.scale.y = 0.4
        self.local_target_marker.scale.z = 0.4
        self.local_target_marker.color.a = 1
        self.local_target_marker.color.r = 1
        self.local_target_marker.color.g = 1
        self.local_target_marker.color.b = 0
        self.local_target_marker.pose.orientation.w = 1.0

    def visualize_local_target(self):
        self.local_target_marker.header.stamp = rospy.Time.now()
        self.local_target_marker.pose.position.x = self.target_state[0][0]
        self.local_target_marker.pose.position.y = self.target_state[0][1]
        self.local_target_marker.pose.position.z = self.target_state[0][2]

        self.local_target_pub.publish(self.local_target_marker)


if __name__ == "__main__":

    traj_planner = TrajPlanner()

    rospy.spin()
