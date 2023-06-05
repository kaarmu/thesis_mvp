#! /usr/bin/env python3

import rospy
from std_srvs.srv import Trigger
from geometry_msgs.msg import PolygonStamped, Point32

from svea_msgs.msg import VehicleState as VehicleStateMsg

def load_param(name, value=None):
    if value is None:
        assert rospy.has_param(name), f'Missing parameter "{name}"'
    return rospy.get_param(name, value)

def isinside(pt, box):
    xl, yl, xu, yu = box
    x, y = pt
    return (xl < x < xu) and (yl < y < yu)

class geofence:

    WARN_MESSAGE = 'Geofence for "%s" was triggered but client was not successful: %s'

    def __init__(self):

        ## Initialize node

        rospy.init_node('geofence')

        ## Parameters

        self.BOX = load_param('~box')
        self.STATE_TOP = load_param('~state')
        self.TRIGGER_SRV = load_param('~trigger')

        ## Services

        rospy.wait_for_service(self.TRIGGER_SRV)
        rospy.loginfo('trigger service ready')

        self.trigger = rospy.ServiceProxy(self.TRIGGER_SRV, Trigger)

        ## Topics

        self.pub_fenced_area = rospy.Publisher(
            'fenced_area',
            PolygonStamped,
            queue_size=1,
            latch=True,
        )
        self.publish_fenced_area()

        self.sub_state = rospy.Subscriber(
            self.STATE_TOP,
            VehicleStateMsg,
            self.state_cb,
        )

        rospy.loginfo('Geofence is up!')

    def run(self):
        rospy.spin()

    def publish_fenced_area(self):
        xl, yl, xu, yu = self.BOX

        msg = PolygonStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = rospy.Time.now()
        msg.polygon.points.append(Point32(xl, yl, 0))
        msg.polygon.points.append(Point32(xu, yl, 0))
        msg.polygon.points.append(Point32(xu, yu, 0))
        msg.polygon.points.append(Point32(xl, yu, 0))

        self.pub_fenced_area.publish(msg)

    def state_cb(self, msg):
        pt = (msg.x, msg.y)
        if isinside(pt, self.BOX):
            resp = self.trigger()
            if not resp.success:
                rospy.logwarn(self.WARN_MESSAGE, self.STATE_TOP, resp.message)

if __name__ == '__main__':

    geofence().run()

