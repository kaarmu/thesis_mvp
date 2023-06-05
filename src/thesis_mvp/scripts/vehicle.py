#! /usr/bin/env python3

import numpy as np

import rospy
from std_msgs.msg import Empty
from geometry_msgs.msg import Point, PointStamped
from svea_msgs.msg import VehicleState
from svea.controllers.pure_pursuit import PurePursuitController
from svea.models.bicycle import SimpleBicycleModel
from svea.simulators.sim_SVEA import SimSVEA
from svea.interfaces import ActuationInterface, LocalizationInterface
from svea.data import RVIZPathHandler

from thesis_mvp.srv import Path as PathService, PathRequest, PathResponse
from thesis_mvp.track import Arc, Track

def load_param(name, value=None):
    if value is None:
        assert rospy.has_param(name), f'Missing parameter "{name}"'
    return rospy.get_param(name, value)


class vehicle:

    DELTA_TIME = 0.01
    TRAJ_LEN = 10
    TARGET_VELOCITY = 0.5
    RATE = 1e9

    def __init__(self):

        ## Initialize node

        rospy.init_node('vehicle')

        ## Parameters

        self.NAME = load_param('~name')
        self.GOAL = load_param('~goal')
        self.IS_SIM = load_param('~is_sim', False)
        self.USE_RVIZ = load_param('~use_rviz', False)
        self.SIM_START = load_param('~sim_start', [])

        self.GOAL_POINTS = {
            'A': Point(0.0, -2.5, 0),
            'B': Point(0.0, -0.5, 0),
            'C': Point(0.0, +2.0, 0),
        }

        ## Start interfaces

        self.rate = rospy.Rate(10)

        self.actuation = ActuationInterface().start(wait=True)
        self.localization = LocalizationInterface().start()

        self.state = self.localization.state
        self.localization.add_callback(lambda state: setattr(self, 'state', state))

        ## Create simulators, models, managers, etc.

        self.controller = PurePursuitController()
        self.set_velocity(self.TARGET_VELOCITY)

        if self.USE_RVIZ:

            self.rviz = RVIZPathHandler()
            self.localization.add_callback(self.rviz.log_state)

        if self.IS_SIM:

            state_pub = rospy.Publisher('/state', VehicleState, queue_size=10)
            rospy.Timer(rospy.Duration(1) / 10,
                        lambda *_: state_pub.publish(self.state.state_msg))

            # simulator need a model to simulate
            self.sim_model = SimpleBicycleModel()
            (self.sim_model.state.x,
             self.sim_model.state.y,
             self.sim_model.state.yaw) = self.SIM_START

            # start the simulator immediately, but paused
            self.simulator = SimSVEA(self.sim_model,
                                     dt=0.01,
                                     run_lidar=True,
                                     start_paused=True).start()

        ## Subscriber

        
        rospy.loginfo('Waiting for LTMS')
        rospy.wait_for_message('/ltms/heartbeat', Empty)
        rospy.loginfo("Connected to LTMS")

        self.goal_pub = rospy.Publisher('/goal', PointStamped, queue_size=10)
        self.request_path = rospy.ServiceProxy('/ltms/request_path', PathService)
        rospy.Timer(rospy.Duration(3),
                    lambda *_: self.path_requester())

        ## Start simulation

        if self.IS_SIM:

            # everything ready to go -> unpause simulator
            self.simulator.toggle_pause_simulation()

        self.set_goal(self.GOAL)

        print(f"Starting going to {self.goal_name}")

    def set_velocity(self, velocity):
        self.controller.target_velocity = velocity

    def set_goal(self, goal_name):
        self.goal_name = goal_name
        self.goal = self.GOAL_POINTS[goal_name]

        msg = PointStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = rospy.Time.now()
        msg.point = self.goal
        self.goal_pub.publish(msg)

    def path_requester(self):
        req = PathRequest()
        req.name = self.NAME
        req.goal_name = self.goal_name
        req.goal = self.goal

        resp: PathResponse = self.request_path(req)

        if resp.path:

            self.path = np.array([(p.pose.position.x, p.pose.position.y) 
                                  for p in resp.path.poses])

            if self.USE_RVIZ and self.path.any():
                xtraj, ytraj = zip(*self.path)
                self.rviz.update_traj(xtraj, ytraj)
        
        self.set_velocity(resp.velocity)

        return self.path

    def dist_to(self, xy):
        xy = np.asarray(xy)
        state = np.array([self.state.x, self.state.y])
        return np.linalg.norm(xy - state, axis=-1)

    def run(self):

        while self.keep_alive():

            path = self.path_requester()
            i = self.dist_to(path).argmin()

            if len(path) - 10 < i:
                i = 0

            for point in path[i:]:

                if self.path is not path:
                    break

                self.target = tuple(point)
                while 0.4 < self.dist_to(self.target):
                    if not self.keep_alive(): return
                    self.rate.sleep()
                    self.spin()

                if self.dist_to((self.goal.x, self.goal.y)) < 0.4:
                    self.set_goal('A' if self.goal_name == self.GOAL else
                                  self.GOAL)
                    break

    def keep_alive(self):
        return not rospy.is_shutdown()

    def spin(self):

        steering, velocity = self.controller.compute_control(self.state,
                                                             self.target)
        self.actuation.send_control(steering, velocity)

        if self.USE_RVIZ:
            self.rviz.update_target(self.target)
            self.rviz.visualize_data()

if __name__ == '__main__':

    ## Start node ##

    vehicle().run()
