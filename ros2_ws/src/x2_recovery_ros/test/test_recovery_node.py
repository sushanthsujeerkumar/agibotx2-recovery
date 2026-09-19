from unittest.mock import patch

import pytest
import rclpy
from rclpy.parameter import Parameter
from std_srvs.srv import Trigger

from x2_recovery_ros.recovery_node import RecoveryNode


@pytest.fixture
def node():
    rclpy.init()
    instance = RecoveryNode()
    try:
        yield instance
    finally:
        instance.destroy_node()
        rclpy.shutdown()


def test_acceptance_is_queued_before_any_simulator_work(node):
    with patch.object(node._mp_context, 'Process') as process:
        response = node._start_callback(Trigger.Request(), Trigger.Response())
        assert response.success
        assert node._pending is not None
        assert node._busy
        process.assert_not_called()
        second = node._start_callback(Trigger.Request(), Trigger.Response())
        assert not second.success


def test_timeout_fails_without_waiting_for_simulator(node):
    node._start_callback(Trigger.Request(), Trigger.Response())
    with patch('x2_recovery_ros.recovery_node.time.monotonic', return_value=node._started_at + 61.0):
        node._tick()
    assert node._status == 'FAILED'
    assert not node._busy
    assert node._pending is None
    assert node._start_callback(Trigger.Request(), Trigger.Response()).success


def test_policy_requires_checkpoint(node):
    node.set_parameters([Parameter('controller', value='policy')])
    response = node._start_callback(Trigger.Request(), Trigger.Response())
    assert not response.success
    assert not node._busy


@pytest.mark.parametrize('value', [0.0, -1.0, float('inf'), float('nan')])
def test_invalid_timeout_rejected(node, value):
    node.set_parameters([Parameter('timeout_s', value=value)])
    response = node._start_callback(Trigger.Request(), Trigger.Response())
    assert not response.success


def test_joint_message_uses_simulator_time(node):
    with patch.object(node._joints_pub, 'publish') as publish:
        node._publish_frame({'sim_time': 3.125, 'joint_names': ['hip'], 'joint_positions': [0.4]})
        message = publish.call_args[0][0]
    assert message.header.stamp.sec == 3
    assert message.header.stamp.nanosec == 125000000
    assert message.name == ['hip']
    assert list(message.position) == [0.4]
