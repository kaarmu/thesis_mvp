#! /usr/bin/env python3

from threading import Event

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
from thesis_mvp.controllers.mpc import ModelPredictiveController
from thesis_mvp.models.mpc.bicycle import BicycleModel

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
            'A': Point(0.0, -2.0, 0),
            'B': Point(0.0, +0.0, 0),
            'C': Point(0.0, +2.0, 0),
        }

        ## Start interfaces

        self.rate = rospy.Rate(10)

        self.actuation = ActuationInterface().start(wait=True)
        self.localization = LocalizationInterface().start()

        self.state = self.localization.state
        self.localization.add_callback(lambda state: setattr(self, 'state', state))

        ## Create simulators, models, managers, etc.

        self.USE_MPC = True

        if self.USE_MPC:
            mpc_model = BicycleModel()
            mpc_controller = ModelPredictiveController(
                mpc_model, 
                step=0.1, 
                horizon=5,
                weights = {'input': {'steering': 1, 'velocity': 1}, 
                           'state': {'x': 1, 'y': 1, 'yaw': 0, 'v': 1},
                           'terminal': {'x': 10, 'y': 10, 'yaw': 0, 'v': 10}},
                constraints = {('upper', 'steering'): +np.pi/5,
                               ('lower', 'steering'): -np.pi/5,
                               ('upper', 'velocity'): +0.8,
                               ('lower', 'velocity'): -0.5},
            )
            mpc_controller.set_initial_guess()
            self.controller = mpc_controller

        else:
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
        # rospy.wait_for_message('/ltms/heartbeat', Empty)
        rospy.loginfo("Connected to LTMS")

        self.path = None
        self.path_name = 'unknown'
        self.path_event = Event()

        if True:
            from nav_msgs.msg import Path
            from geometry_msgs.msg import PoseStamped
            from thesis_mvp.track import Track, Arc

            INTERSECTION_1 = [+2.5, -1.0, +np.pi/2]
            INTERSECTION_2 = [-2.5, -1.0, -np.pi/2] 

            SHARED_CIRCUIT = [
                [1.0, 90],
                [3.0],
                [1.0, 90],
            ]

            SMALL_CIRCUIT = [
                [1.0, 90],
                [3.0],
                [1.0, 90],
            ]

            LARGE_CIRCUIT = [
                [2.5],
                [1.0, 90],
                [3.0],
                [1.0, 90],
                [2.5],
            ]

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

            large_track = Track([
                Arc(*arc) if len(arc) == 1 else
                Arc.from_circle_segment(*arc)
                for arc in LARGE_CIRCUIT
            ], *INTERSECTION_1, POINT_DENSITY=100)

            shared_track = Track([
                Arc(*arc) if len(arc) == 1 else
                Arc.from_circle_segment(*arc)
                for arc in SHARED_CIRCUIT
            ], *INTERSECTION_2, POINT_DENSITY=100)
            # shared_track.connects_to(large_track)

            small_track = Track([
                Arc(*arc) if len(arc) == 1 else
                Arc.from_circle_segment(*arc)
                for arc in SMALL_CIRCUIT
            ], *INTERSECTION_1, POINT_DENSITY=100)
            small_track.connects_to(shared_track)


            self.path = np.array([(p.pose.position.x, p.pose.position.y) 
                                  for p in to_path(small_track).poses])
            self.path_event.set()

            if self.USE_RVIZ and self.path.any():
                xtraj, ytraj = zip(*self.path)
                self.rviz.update_traj(xtraj, ytraj)

        self.goal_pub = rospy.Publisher('/goal', PointStamped, queue_size=10)
        self.request_path = rospy.ServiceProxy('/ltms/request_path', PathService)
        rospy.Timer(rospy.Duration(2),
                    lambda *_: self.path_requester())

        ## Start simulation

        if self.IS_SIM:

            # everything ready to go -> unpause simulator
            self.simulator.toggle_pause_simulation()

        self.set_goal(self.GOAL)

        print('Starting!')

    def set_velocity(self, velocity):
        self.target_velocity = velocity
        if not self.USE_MPC:
            self.controller.target_velocity = velocity

    def set_goal(self, goal_name):
        self.goal_name = goal_name
        self.goal = self.GOAL_POINTS[goal_name]

        msg = PointStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = rospy.Time.now()
        msg.point = self.goal
        self.goal_pub.publish(msg)

        rospy.loginfo(f'Going to {self.goal_name}')

    def path_requester(self):
        return
        req = PathRequest()
        req.name = self.NAME
        req.state = self.state.state_msg
        req.goal_name = self.goal_name
        req.goal = self.goal

        print(f'> Path requested for {self.goal_name}')
        resp: PathResponse = self.request_path(req)
        print(f'=> Path received')

        if resp.path_name != self.path_name and resp.path_name != 'unknown':

            self.path_name = resp.path_name

            self.path = np.array([(p.pose.position.x, p.pose.position.y) 
                                  for p in resp.path.poses])

            if self.USE_RVIZ and self.path.any():
                xtraj, ytraj = zip(*self.path)
                self.rviz.update_traj(xtraj, ytraj)

            self.path_event.set()
            print(f'=> Path updated from {self.path_name} to {resp.path_name}')
        
        self.set_velocity(resp.velocity)

    def dist_to(self, xy):
        xy = np.asarray(xy)
        state = np.array([self.state.x, self.state.y])
        return np.linalg.norm(xy - state, axis=-1)

    def run(self):

        self.path_event.wait()
        
        while self.keep_alive():

            path = self.path
            i = self.dist_to(path).argmin()

            for point in path[i:]:

                if self.path_event.is_set():
                    self.path_event.clear()
                    break

                self.target = tuple(point)
                while 0.4 < self.dist_to(self.target) < 1.0:
                    if not self.keep_alive(): return
                    self.rate.sleep()
                    self.spin()

                if self.dist_to((self.goal.x, self.goal.y)) < 0.4:
                    self.set_goal('A' if self.goal_name == self.GOAL else
                                  self.GOAL)

    def keep_alive(self):
        return not rospy.is_shutdown()

    def spin(self):

        steering, velocity = 0, 0
        if self.USE_MPC:
            x0 = np.array([self.target[0] - self.state.x,
                           self.target[1] - self.state.y,
                           self.target_velocity - self.state.v,
                           self.state.yaw])
            steering, velocity = self.controller.make_step(x0)
            print(steering, velocity)
        else:
            steering, velocity = self.controller.compute_control(self.state,
                                                                 self.target)
        self.actuation.send_control(steering, velocity)

        if self.USE_RVIZ:
            self.rviz.update_target(self.target)
            self.rviz.visualize_data()

if __name__ == '__main__':

    ## Start node ##

    vehicle().run()
