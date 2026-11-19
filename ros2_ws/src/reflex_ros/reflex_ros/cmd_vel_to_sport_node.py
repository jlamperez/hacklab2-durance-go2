"""Drive the Go2 with any standard ROS 2 teleop: /cmd_vel in, sport-mode Move out.

  subscribes  /cmd_vel              geometry_msgs/Twist  (teleop_twist_keyboard, a joystick, Nav2...)
  publishes   /api/sport/request    unitree_api/Request  (Move, or StopMove for a zero twist)

The same node drives the simulated dog and, unchanged, a real Go2 in sport mode.
Speeds are clamped here. With `timeout_s` > 0, StopMove is sent if /cmd_vel goes quiet:
set it on the real robot, and make the teleop repeat its command.
"""
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

from reflex.runtime import sport_api
from unitree_api.msg import Request


class CmdVelToSport(Node):
    def __init__(self):
        super().__init__("cmd_vel_to_sport")
        p = self.declare_parameter
        self.max = (float(p("max_vx", 0.8).value), float(p("max_vy", 0.4).value), float(p("max_wz", 0.9).value))
        self.timeout_s = float(p("timeout_s", 0.0).value)
        self.pub = self.create_publisher(Request, sport_api.SPORT_REQUEST_TOPIC, 10)
        self.create_subscription(Twist, "/cmd_vel", self.on_twist, 10)
        self.create_timer(0.1, self.on_timer)
        self._last_t, self._moving = None, False

    def send(self, cmd, param: str = "") -> None:
        req = Request()
        req.header.identity.api_id = cmd.api_id
        req.parameter = param
        self.pub.publish(req)

    def on_twist(self, msg: Twist) -> None:
        raw = (msg.linear.x, msg.linear.y, msg.angular.z)
        vx, vy, wz = (max(-m, min(m, v)) for v, m in zip(raw, self.max))
        self._last_t = self.get_clock().now().nanoseconds * 1e-9
        if vx == 0.0 and vy == 0.0 and wz == 0.0:
            self.send(sport_api.STOP_MOVE)
            self._moving = False
        else:
            self.send(sport_api.MOVE, sport_api.move_params(vx, vy, wz))
            self._moving = True

    def on_timer(self) -> None:
        if self.timeout_s > 0 and self._moving and \
                self.get_clock().now().nanoseconds * 1e-9 - self._last_t > self.timeout_s:
            self.get_logger().warn("/cmd_vel is silent: sending StopMove")
            self.send(sport_api.STOP_MOVE)
            self._moving = False


def main():
    rclpy.init()
    try:
        rclpy.spin(CmdVelToSport())
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass


if __name__ == "__main__":
    main()
