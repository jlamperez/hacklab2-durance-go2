"""Send the Go2's built-in tricks from the keyboard, as sport-mode requests.

  publishes   /api/sport/request   unitree_api/Request
  subscribes  /reflex/trick        std_msgs/String   (a command name, e.g. "Hello"; for scripts)

    pixi run -e ros ros-tricks                          # gestures only
    pixi run -e ros ros-tricks allow_acrobatics:=true   # also jumps and flips

Gestures:  h Hello   s Sit   u RiseSit (get up)   t Stretch   c Content   1 Dance1   2 Dance2
           p Scrape  e Heart   SPACE StopMove   b BalanceStand   r RecoveryStand   q quit
Acrobatics (only with allow_acrobatics:=true; the robot leaves the ground, clear the area):
           J FrontJump   P FrontPounce   F FrontFlip   L LeftFlip   B BackFlip   (capital letters)
Always:    ! Damp (motors limp: the robot drops; emergency only)

The simulator imitates Sit, RiseSit and Hello and ignores the rest. On the real robot
the trick is Unitree's own; which ones a given Go2 accepts is not known.
"""
import sys
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from reflex.runtime import sport_api as api
from unitree_api.msg import Request

KEYS = {"h": api.HELLO, "s": api.SIT, "u": api.RISE_SIT, "t": api.STRETCH, "c": api.CONTENT, "1": api.DANCE1,
        "2": api.DANCE2, "p": api.SCRAPE, "e": api.HEART, " ": api.STOP_MOVE, "b": api.BALANCE_STAND,
        "r": api.RECOVERY_STAND, "!": api.DAMP,
        "J": api.FRONT_JUMP, "P": api.FRONT_POUNCE, "F": api.FRONT_FLIP, "L": api.LEFT_FLIP, "B": api.BACK_FLIP}
BY_NAME = {c.name.lower(): c for c in api.ALL}


class Tricks(Node):
    def __init__(self):
        super().__init__("reflex_tricks")
        self.allow_acrobatics = bool(self.declare_parameter("allow_acrobatics", False).value)
        self.pub = self.create_publisher(Request, api.SPORT_REQUEST_TOPIC, 10)
        self.create_subscription(String, "/reflex/trick", lambda m: self.send_named(m.data), 10)
        self.say("h Hello  s Sit  u RiseSit  t Stretch  c Content  1/2 Dance  p Scrape  e Heart  "
                 "SPACE StopMove  b BalanceStand  r RecoveryStand  ! Damp  q quit")
        self.say("acrobatics ENABLED: J FrontJump  P FrontPounce  F FrontFlip  L LeftFlip  B BackFlip"
                 if self.allow_acrobatics else "acrobatics disabled (allow_acrobatics:=true to enable)")

    def say(self, text: str) -> None:
        print(f"\r[tricks] {text}\r", flush=True)

    def send(self, cmd) -> bool:
        if cmd in api.ACROBATICS and not self.allow_acrobatics:
            self.say(f"{cmd.name} refused: acrobatics are disabled")
            return False
        req = Request()
        req.header.identity.api_id = cmd.api_id
        self.pub.publish(req)
        self.say(f"sent {cmd.name} (api_id {cmd.api_id})")
        return True

    def send_named(self, name: str) -> None:
        cmd = BY_NAME.get(name.strip().lower())
        if cmd is None or cmd in (api.MOVE, api.EULER, api.POSE):   # these need parameters
            self.say(f"unknown trick {name!r}")
            return
        self.send(cmd)


def keyboard(node: Tricks) -> None:
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while rclpy.ok():
            key = sys.stdin.read(1)
            if key == "q":
                rclpy.shutdown()
                return
            cmd = KEYS.get(key)
            if cmd is not None:
                node.send(cmd)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main():
    rclpy.init()
    node = Tricks()
    if sys.stdin.isatty():
        threading.Thread(target=keyboard, args=(node,), daemon=True).start()
    else:
        node.say("no keyboard (not a terminal): send names on /reflex/trick")
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass


if __name__ == "__main__":
    main()
