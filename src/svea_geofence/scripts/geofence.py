#! /usr/bin/env python3

import rospy
from std_srvs.srv import Trigger
from geometry_msgs.msg import PolygonStamped, Point32

from rosonic import Node, Parameter
from svea_msgs.msg import VehicleState as VehicleStateMsg

def isinside(pt, box):
    xl, yl, xu, yu = box
    x, y = pt
    return (xl < x < xu) and (yl < y < yu)

class geofence(Node):

    BOX = Parameter('~box')
    STATE_TOP = Parameter('~state')
    TRIGGER_SRV = Parameter('~trigger')

    WARN_MESSAGE = 'Geofence for "%s" was triggered but client was not successful: %s'

    def __init__(self):

        rospy.wait_for_service(self.TRIGGER_SRV)
        self.log('trigger service ready')

        self.trigger = rospy.ServiceProxy(self.TRIGGER_SRV, Trigger)

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

        self.log('geofence is up!')

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
            resp.success or self.logwarn(self.WARN_MESSAGE, self.STATE_TOP, resp.message)

if __name__ == '__main__':

    geofence().run()
