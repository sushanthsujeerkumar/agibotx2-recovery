"""Log the current status and a joint position from actual simulator messages."""

import signal

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String


class TelemetryNode(Node):
    def __init__(self):
        super().__init__('x2_telemetry')
        self.declare_parameter('log_period_s', 1.0)
        period = float(self.get_parameter('log_period_s').value)
        if period <= 0.0:
            raise ValueError('log_period_s must be positive')
        self._status = 'UNKNOWN'
        self._joint = None
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)
        self._status_sub = self.create_subscription(String, '/x2/recovery_status', self._on_status, qos)
        self._joints_sub = self.create_subscription(JointState, '/x2/joint_states', self._on_joints, 10)
        self._timer = self.create_timer(period, self._log)

    def _on_status(self, message):
        if message.data != self._status:
            self._status = message.data
            if self._status == 'RUNNING':
                self._joint = None
            self.get_logger().info(f'Status: {self._status}')

    def _on_joints(self, message):
        if message.name and message.position:
            self._joint = (message.name[0], message.position[0],
                           message.header.stamp.sec + message.header.stamp.nanosec * 1e-9)

    def _log(self):
        if self._joint is None:
            self.get_logger().info(f'status={self._status}; awaiting simulator joint telemetry')
        else:
            name, position, sim_time = self._joint
            self.get_logger().info(
                f'status={self._status}; {name}={position:.5f} rad; sim_time={sim_time:.3f} s'
            )


def main(args=None):
    rclpy.init(args=args)
    node = TelemetryNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
