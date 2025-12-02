import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TwistStamped

class TwistToTwistStamped(Node):
    def __init__(self):
        super().__init__('twist_to_twiststamped')
        self.sub = self.create_subscription(Twist, '/cmd_vel', self.cb, 10)
        self.pub = self.create_publisher(TwistStamped, '/mavros/setpoint_velocity/cmd_vel', 10)
        self.t0 = TwistStamped()
        self.ts = Twist()
        self.timer = self.create_timer(0.05, self.pub_messsge)

    def cb(self, msg:Twist):
        self.ts = msg

    def pub_messsge(self):
        self.t0.twist = self.ts
        self.pub.publish(self.t0)

def main(args=None):
    rclpy.init(args=args)
    node = TwistToTwistStamped()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
