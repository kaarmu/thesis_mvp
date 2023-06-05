#! /usr/bin/env python3

import numpy as np

import rospy
from std_msgs.msg import Empty
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from svea_msgs.msg import VehicleState

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
            [1.0],
            [0.5, 90],
            [4.0],
            [0.5, 90],
            [1.0],
        ]

        self.SMALL_CIRCUIT = [
            [0.5, 90],
            [4.0],
            [0.5, 90],
        ]

        self.LARGE_CIRCUIT = [
            [2.5],
            [0.5, 90],
            [4.0],
            [0.5, 90],
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
        self.small_track.connects_to(self.shared_track)

        self.large_track = Track([
            Arc(*arc) if len(arc) == 1 else
            Arc.from_circle_segment(*arc)
            for arc in self.LARGE_CIRCUIT
        ], *self.INTERSECTION_1, POINT_DENSITY=100)
        self.large_track.connects_to(self.shared_track)

        ## Subscribers

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

    def request_path_srv_cb(self, req: PathRequest) -> PathResponse:
        name = req.name
        goal_name = req.goal_name

        state = self.clients[name]
        self.shared_track.update_road_user(name, state)
        self.small_track.update_road_user(name, state)
        self.large_track.update_road_user(name, state)

        in_shared_track = name in self.shared_track.road_users
        in_small_track = name in self.small_track.road_users
        in_large_track = name in self.large_track.road_users
        in_intersection_1 = 1.0 >= np.hypot(self.INTERSECTION_1[0] - state.x,
                                            self.INTERSECTION_1[1] - state.y)
        in_intersection_2 = 1.0 >= np.hypot(self.INTERSECTION_2[0] - state.x,
                                            self.INTERSECTION_2[1] - state.y)

        resp = PathResponse()
        if in_intersection_1:
            # is on both small and large track
            resp.path = (
                self.to_path(self.small_track) if goal_name == 'A' else
                self.to_path(self.small_track) if goal_name == 'B' else
                self.to_path(self.large_track) if goal_name == 'C' else
                []
            )
        elif in_intersection_2:
            # is on both small and large track
            resp.path = self.to_path(self.shared_track)
        elif in_shared_track:
            resp.path = self.to_path(self.shared_track)
        elif in_small_track:
            # is on small track
            resp.path = (
                self.to_path(self.small_track) if goal_name == 'A' else
                self.to_path(self.small_track) if goal_name == 'B' else
                self.to_path(self.small_track) if goal_name == 'C' else
                []
            )
        elif in_large_track:
            # is on large track
            resp.path = (
                self.to_path(self.large_track) if goal_name == 'A' else
                self.to_path(self.large_track) if goal_name == 'B' else
                self.to_path(self.large_track) if goal_name == 'C' else
                []
            )

        if in_large_track and len(self.large_track.road_users) <= 1:
            resp.velocity = 1.0
        else:
            resp.velocity = 0.5

        return resp

    def run(self):
        rospy.spin()

if __name__ == '__main__':

    ## Start node ##

    ltms().run()
