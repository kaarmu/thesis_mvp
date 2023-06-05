#! /usr/bin/env python3

import numpy as np

import rospy
from svea.controllers.pure_pursuit import PurePursuitController
from svea.models.bicycle import SimpleBicycleModel
from svea.simulators.sim_SVEA import SimSVEA
from svea.interfaces import ActuationInterface, LocalizationInterface
from svea.data import RVIZPathHandler

from thesis_mvp.track import Arc, Track

def load_param(name, value=None):
    if value is None:
        assert rospy.has_param(name), f'Missing parameter "{name}"'
    return rospy.get_param(name, value)


class draft:

    DELTA_TIME = 0.01
    TRAJ_LEN = 10
    TARGET_VELOCITY = 0.6
    RATE = 1e9

    def __init__(self):

        ## Initialize node

        rospy.init_node('pure_pursuit')

        ## Parameters

        self.ARCS = load_param('~arcs')
        self.TRACK_START = load_param('~track_start')
        self.IS_SIM = load_param('~is_sim', False)
        self.USE_RVIZ = load_param('~use_rviz', False)

        ## Start interfaces

        self.rate = rospy.Rate(10)

        self.actuation = ActuationInterface().start(wait=True)
        self.localization = LocalizationInterface().start()

        self.state = self.localization.state
        self.localization.add_callback(lambda state: setattr(self, 'state', state))

        ## Create simulators, models, managers, etc.

        self.track = Track([
            Arc(*arc) if len(arc) == 1 else
            Arc.from_circle_segment(*arc)
            for arc in self.ARCS
        ], *self.TRACK_START, POINT_DENSITY=100)
        self.track.connects_to(self.track)

        self.controller = PurePursuitController()
        self.controller.target_velocity = self.TARGET_VELOCITY

        if self.USE_RVIZ:

            self.rviz = RVIZPathHandler()
            self.localization.add_callback(self.rviz.log_state)

        if self.IS_SIM:

            # simulator need a model to simulate
            self.sim_model = SimpleBicycleModel()

            # start the simulator immediately, but paused
            self.simulator = SimSVEA(self.sim_model,
                                     dt=0.01,
                                     run_lidar=True,
                                     start_paused=True).start()

        # everything ready to go -> unpause simulator
        if self.IS_SIM:
            self.simulator.toggle_pause_simulation()

    def run(self):
        xtraj, ytraj = self.track.cartesian
        if self.USE_RVIZ:
            self.rviz.update_traj(xtraj, ytraj)
        for target in self.track:
            xt, yt = self.target = target
            dist = lambda: np.hypot(xt - self.state.x,
                                    yt - self.state.y)
            while 0.2 < dist():
                if not self.keep_alive(): return
                self.rate.sleep()
                self.spin()

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

    draft().run()
