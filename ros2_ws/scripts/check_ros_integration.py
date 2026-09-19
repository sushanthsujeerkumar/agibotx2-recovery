#!/usr/bin/env python3
"""Check a running real-simulator ROS launch; produce machine-readable evidence.

No simulator doubles are used here. Start recovery.launch.py separately, then run
this probe from a terminal with the same ROS_DOMAIN_ID and sourced workspace.
"""

import argparse
import json
import math
from pathlib import Path
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expect-final', choices=['SUCCEEDED', 'FAILED'], default='FAILED')
    parser.add_argument('--deadline', type=float, default=90.0)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    rclpy.init()
    node = Node('x2_integration_probe')
    statuses = []
    frames = []

    def on_status(message):
        if not statuses or statuses[-1] != message.data:
            statuses.append(message.data)

    def on_joints(message):
        frames.append({
            'names': list(message.name), 'positions': list(message.position),
            'sim_time': message.header.stamp.sec + message.header.stamp.nanosec * 1e-9,
            'received_at': time.monotonic(),
        })

    qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     reliability=ReliabilityPolicy.RELIABLE)
    status_sub = node.create_subscription(String, '/x2/recovery_status', on_status, qos)
    joint_sub = node.create_subscription(JointState, '/x2/joint_states', on_joints, 100)
    client = node.create_client(Trigger, '/x2/start_recovery')
    evidence = {'expected_final_status': args.expect_final}
    try:
        if not client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError('Start service was not available within 10 seconds')
        # Allow DDS subscriptions to match before starting the episode.
        until = time.monotonic() + 0.5
        while time.monotonic() < until:
            rclpy.spin_once(node, timeout_sec=0.05)
        statuses.clear()
        frames.clear()
        started_at = time.monotonic()
        first = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(node, first, timeout_sec=3.0)
        if not first.done():
            raise AssertionError('Acceptance response took longer than 3 seconds')
        first_response_at = time.monotonic()
        evidence['acceptance_seconds'] = first_response_at - started_at
        evidence['first_response'] = {'success': first.result().success, 'message': first.result().message}
        assert first.result().success, evidence['first_response']
        second = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(node, second, timeout_sec=3.0)
        assert second.done(), 'Busy rejection took longer than 3 seconds'
        evidence['second_response'] = {'success': second.result().success, 'message': second.result().message}
        assert not second.result().success, 'Second request was accepted while busy'
        while time.monotonic() - started_at < args.deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if 'RUNNING' in statuses and statuses[-1] in ('SUCCEEDED', 'FAILED'):
                break
        evidence['statuses'] = statuses
        evidence['elapsed_seconds'] = time.monotonic() - started_at
        evidence['joint_frame_count'] = len(frames)
        assert 'RUNNING' in statuses, 'RUNNING was never observed'
        assert statuses[-1] == args.expect_final, evidence['statuses']
        assert len(frames) >= 2, 'Fewer than two actual simulator joint frames received'
        assert frames[0]['received_at'] >= first_response_at, 'A joint frame preceded acceptance'
        for frame in frames:
            assert len(frame['names']) == len(frame['positions']) > 0
            assert len(set(frame['names'])) == len(frame['names'])
            assert all(math.isfinite(value) for value in frame['positions'])
        assert all(b['sim_time'] > a['sim_time'] for a, b in zip(frames, frames[1:])), 'Simulation timestamps did not advance'
        evidence['joint_names'] = frames[0]['names']
        evidence['first_joint_frame'] = {key: value for key, value in frames[0].items() if key != 'received_at'}
        evidence['last_joint_frame'] = {key: value for key, value in frames[-1].items() if key != 'received_at'}
        evidence['positions_changed'] = frames[0]['positions'] != frames[-1]['positions']
        evidence['passed'] = True
    except Exception as error:
        evidence['passed'] = False
        evidence['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        evidence.setdefault('statuses', statuses)
        evidence.setdefault('joint_frame_count', len(frames))
        text = json.dumps(evidence, indent=2)
        print(text)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + '\n')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
