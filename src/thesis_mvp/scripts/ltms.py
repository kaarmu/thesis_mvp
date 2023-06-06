#! /usr/bin/env python3

import numpy as np

import rospy
import tf2_ros
from tf2_geometry_msgs import do_transform_point
from std_msgs.msg import Empty
from nav_msgs.msg import Path
from geometry_msgs.msg import PointStamped, PoseStamped
from svea_msgs.msg import VehicleState
from rsu_msgs.msg import StampedObjectPoseArray

from thesis_mvp.srv import Path as PathService, PathRequest, PathResponse
from thesis_mvp.track import Track, Arc

from tf_conversions import Quaternion, transformations


def to_quaternion(x=0, y=0, z=0):
    x, y, z, w = transformations.quaternion_from_euler(x, y, z)
    return Quaternion(x, y, z, w)

def load_param(name, value=None):
    if value is None:
        assert rospy.has_param(name), f'Missing parameter "{name}"'
    return rospy.get_param(name, value)


class ltms:

    DELTA_TIME = 0.01
    TRAJ_LEN = 10
    TARGET_VELOCITY = 0.6
    RATE = 1e9

    def __init__(self):

        ## Initialize node

        rospy.init_node('ltms')

        ## Parameters

        self.CLIENTS = load_param('~clients')

        self.INTERSECTION_1 = [+2.5, -1.0, +np.pi/2]
        self.INTERSECTION_2 = [-2.5, -1.0, -np.pi/2] 

        self.SHARED_CIRCUIT = [
            [1.0, 90],
            [3.0],
            [1.0, 90],
        ]

        self.SMALL_CIRCUIT = [
            [1.0, 90],
            [3.0],
            [1.0, 90],
        ]

        self.LARGE_CIRCUIT = [
            [2.5],
            [1.0, 90],
            [3.0],
            [1.0, 90],
            [2.5],
        ]

        ## Create simulators, models, managers, etc.

        self.shared_track = Track([
            Arc(*arc) if len(arc) == 1 else
            Arc.from_circle_segment(*arc)
            for arc in self.SHARED_CIRCUIT
        ], *self.INTERSECTION_2, POINT_DENSITY=100)

        self.small_track = Track([
            Arc(*arc) if len(arc) == 1 else
            Arc.from_circle_segment(*arc)
            for arc in self.SMALL_CIRCUIT
        ], *self.INTERSECTION_1, POINT_DENSITY=100)
        # self.small_track.connects_to(self.shared_track)

        self.large_track = Track([
            Arc(*arc) if len(arc) == 1 else
            Arc.from_circle_segment(*arc)
            for arc in self.LARGE_CIRCUIT
        ], *self.INTERSECTION_1, POINT_DENSITY=100)
        # self.large_track.connects_to(self.shared_track)

        ## TF 

        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        ## Subscribers

        self.last_person = rospy.Time(0)

        people_pub = rospy.Publisher('people', PointStamped, queue_size=10)
        def objectposes_cb(msg):
            for objpose in msg.objects:
                if objpose.object.label == 'person':
                    point = PointStamped(msg.header, objpose.pose.pose.position)
                    people_pub.publish(point)
                    point = self.transform_point('map', point)
                    if -0.25 < point.point.x < 0.25 and -0.25 < point.point.y < 0.25: 
                        self.last_person = msg.header.stamp
        rospy.Subscriber('/objectposes', StampedObjectPoseArray, objectposes_cb)

        self.clients = {}
        for client in self.CLIENTS:
            topic = f'/{client}/state'
            pose_pub = rospy.Publisher(f'/{client}/pose', PoseStamped, queue_size=10)

            def state_cb(msg):
                self.clients[client] = msg
                pose = PoseStamped()
                pose.header = msg.header  
                pose.pose.position.x = msg.x 
                pose.pose.position.y = msg.y
                pose.pose.orientation = to_quaternion(z=msg.yaw)
                pose_pub.publish(pose)
            rospy.Subscriber(topic, VehicleState, state_cb)
            rospy.wait_for_message(topic, VehicleState)
            print(f'Connected to {client}')

        ## Services

        rospy.Service('/ltms/request_path', PathService, self.request_path_srv_cb)

        ## Publishers

        heartbeat_pub = rospy.Publisher('/ltms/heartbeat', Empty, queue_size=10)
        heartbeat_pub.publish(Empty())
        rospy.Timer(rospy.Duration(10), lambda *_: heartbeat_pub.publish(Empty()))

    @staticmethod
    def to_path(track):
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = rospy.Time.now()
        for x, y in track:
            pose = PoseStamped() 
            pose.header = path.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            path.poses.append(pose)
        return path

    def transform_point(self, to_frame, point_stamped):
        frame_id = point_stamped.header.frame_id
        trans = self.tfBuffer.lookup_transform(to_frame, frame_id, rospy.Time(0))
        return do_transform_point(point_stamped, trans)

    def request_path_srv_cb(self, req: PathRequest) -> PathResponse:
        name = req.name
        state = req.state 
        goal_name = req.goal_name

        self.shared_track.update_road_user(name, state)
        self.small_track.update_road_user(name, state)
        self.large_track.update_road_user(name, state)

        in_shared_track = name in self.shared_track.road_users
        in_small_track = name in self.small_track.road_users
        in_large_track = name in self.large_track.road_users
        in_intersection_1 = 0.6 >= np.hypot(self.INTERSECTION_1[0] - state.x,
                                            self.INTERSECTION_1[1] - state.y)
        in_intersection_2 = 0.6 >= np.hypot(self.INTERSECTION_2[0] - state.x,
                                            self.INTERSECTION_2[1] - state.y)

        path_name = 'unknown'
        if in_intersection_1:
            # is on both small and large track
            path_name = (
                'small' if goal_name == 'A' else
                'small' if goal_name == 'B' else
                'large' if goal_name == 'C' else
                'unknown'
            )
        elif in_intersection_2:
            # is on both small and large track
            path_name = 'shared'
        elif in_shared_track:
            path_name = 'shared'
        elif in_small_track:
            # is on small track
            path_name = (
                'small' if goal_name == 'A' else
                'small' if goal_name == 'B' else
                'small' if goal_name == 'C' else
                'unknown'
            )
        elif in_large_track:
            # is on large track
            path_name = (
                'large' if goal_name == 'A' else
                'large' if goal_name == 'B' else
                'large' if goal_name == 'C' else
                'unknown'
            )

        velocity = 0.4
        if path_name in ('large', 'shared') and len(self.large_track.road_users) <= 1:
            velocity = 0.6

        time_since_person = rospy.Time.now() - self.last_person
        if path_name == 'small' and time_since_person < rospy.Duration(3):
            path_name = 'large'

        resp = PathResponse()
        resp.velocity = velocity
        resp.path_name = path_name
        resp.path = (
            self.to_path(self.shared_track) if path_name == 'shared' else
            self.to_path(self.small_track) if path_name == 'small' else
            self.to_path(self.large_track) if path_name == 'large' else 
            Path()
        )
        return resp

    def run(self):
        rospy.spin()

if __name__ == '__main__':

    ## Start node ##

    ltms().run()
