"""Accept requests immediately and supervise one isolated simulator episode."""

import math
import multiprocessing
import signal
import time

import rclpy
from builtin_interfaces.msg import Time
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger

from .episode_worker import run_episode


class RecoveryNode(Node):
    def __init__(self, **kwargs):
        super().__init__('x2_recovery', **kwargs)
        for name, default in {
            'controller': 'scripted',
            'checkpoint': '',
            'vendor_assets': '',
            'render': False,
            'seed': 0,
            'timeout_s': 60.0,
            'max_sim_duration_s': 20.0,
            'realtime': True,
            'physics_profile': 'legacy',
        }.items():
            self.declare_parameter(name, default)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)
        self._status_pub = self.create_publisher(String, '/x2/recovery_status', qos)
        self._joints_pub = self.create_publisher(JointState, '/x2/joint_states', 10)
        self._service = self.create_service(Trigger, '/x2/start_recovery', self._start_callback)
        self._mp_context = multiprocessing.get_context('spawn')
        self._process = None
        self._connection = None
        self._pending = None
        self._busy = False
        self._status = 'IDLE'
        self._started_at = 0.0
        self._timeout_s = 0.0
        self._timer = self.create_timer(0.02, self._tick)
        self._heartbeat = self.create_timer(1.0, self._publish_status)
        self._publish_status()
        self.get_logger().info('Ready: /x2/start_recovery; simulator starts after an accepted response')

    def _publish_status(self):
        self._status_pub.publish(String(data=self._status))

    def _set_status(self, status):
        self._status = status
        self._publish_status()
        self.get_logger().info(f'Recovery status: {status}')

    def _start_callback(self, _request, response):
        if self._busy:
            response.success = False
            response.message = 'Busy: a recovery attempt is already queued or running'
            return response
        config = {name: self.get_parameter(name).value for name in (
            'controller', 'checkpoint', 'render', 'seed', 'timeout_s',
            'max_sim_duration_s', 'realtime', 'physics_profile', 'vendor_assets',
        )}
        if config['controller'] not in ('scripted', 'policy', 'reference_residual', 'full_recovery'):
            response.success = False
            response.message = 'controller must be scripted, policy, reference_residual or full_recovery'
            return response
        if config['physics_profile'] not in ('legacy', 'guarded_v2'):
            response.success = False
            response.message = 'physics_profile must be legacy or guarded_v2'
            return response
        if config['controller'] in ('policy', 'reference_residual', 'full_recovery') and not config['checkpoint']:
            response.success = False
            response.message = 'policy/reference controller requires a checkpoint path'
            return response
        if config['controller'] == 'full_recovery' and config['physics_profile'] != 'guarded_v2':
            response.success = False
            response.message = 'full_recovery requires guarded_v2 physics'
            return response
        if config['controller'] == 'reference_residual' and (
                not config['vendor_assets'] or config['physics_profile'] != 'guarded_v2'):
            response.success = False
            response.message = 'reference_residual requires vendor_assets and guarded_v2 physics'
            return response
        for name in ('timeout_s', 'max_sim_duration_s'):
            if not math.isfinite(config[name]) or config[name] <= 0.0:
                response.success = False
                response.message = f'{name} must be finite and positive'
                return response
        # A single-threaded executor sends this response before it can enter _tick.
        # Reserving busy here also rejects a request arriving before worker startup.
        self._busy = True
        self._pending = config
        self._started_at = time.monotonic()
        self._timeout_s = config['timeout_s']
        response.success = True
        response.message = 'Accepted: recovery execution is queued'
        return response

    def _tick(self):
        if not self._busy:
            self._reap_finished_process()
            return
        if time.monotonic() - self._started_at >= self._timeout_s:
            self._finish(False, 'wall-clock timeout (includes simulator startup)')
            return
        if self._pending is not None:
            config, self._pending = self._pending, None
            self._set_status('RUNNING')
            self._reap_finished_process(force=True)
            receiver, sender = self._mp_context.Pipe(duplex=False)
            self._connection = receiver
            try:
                self._process = self._mp_context.Process(
                    target=run_episode, args=(config, sender), daemon=True,
                    name='x2-recovery-simulator',
                )
                self._process.start()
            except Exception as error:
                self._finish(False, f'Cannot start simulator process: {error}')
                return
            finally:
                sender.close()
        try:
            # Bound draining so ROS requests and watchdog always get executor time.
            for _ in range(200):
                if self._connection is None or not self._connection.poll():
                    break
                kind, value = self._connection.recv()
                if kind == 'frame':
                    self._publish_frame(value)
                elif kind == 'result':
                    if value.get('traceback'):
                        self.get_logger().error(value['traceback'])
                    self._finish(value['success'], value['reason'])
                    return
                else:
                    raise ValueError(f'Unknown simulator message: {kind}')
            if self._process is not None and not self._process.is_alive():
                if self._connection is not None and self._connection.poll():
                    return  # Drain any remaining result on the next timer callback.
                self._finish(False, f'Simulator exited without a result (exit={self._process.exitcode})')
        except (EOFError, OSError, ValueError, KeyError) as error:
            self._finish(False, f'Simulator communication error: {error}')

    def _publish_frame(self, frame):
        nanoseconds = round(frame['sim_time'] * 1_000_000_000)
        message = JointState()
        message.header.stamp = Time(sec=nanoseconds // 1_000_000_000,
                                    nanosec=nanoseconds % 1_000_000_000)
        message.name = frame['joint_names']
        message.position = frame['joint_positions']
        self._joints_pub.publish(message)

    def _finish(self, success, reason):
        self._pending = None
        self._busy = False
        self._set_status('SUCCEEDED' if success else 'FAILED')
        self.get_logger().info(f'Recovery completed: {reason}')
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        if self._process is not None and self._process.is_alive():
            # Normal results get a brief opportunity to close the viewer/runtime.
            self._process.join(timeout=0.05)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=0.05)
        self._reap_finished_process(force=True)

    def _reap_finished_process(self, force=False):
        if self._process is None:
            return
        try:
            if force and self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=0.1)
            if not self._process.is_alive():
                self._process.join(timeout=0.0)
                self._process.close()
                self._process = None
        except (ValueError, AssertionError):
            self._process = None

    def destroy_node(self):
        self._pending = None
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        self._reap_finished_process(force=True)
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RecoveryNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # A terminal and launch may both send SIGINT; finish cleanup once.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
