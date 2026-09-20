"""Unit tests use explicit test doubles; production never synthesizes telemetry."""
import sys
from types import ModuleType
from unittest.mock import patch

import pytest

from x2_recovery_ros.episode_worker import checked_frame, run_episode


def frame(**changes):
    return {'joint_names': ['hip'], 'joint_positions': [0.25], 'sim_time': 0.02,
            'success': False, 'invalid': False, 'trajectory_limits': {'ok': True}, **changes}


@pytest.mark.parametrize('changes', [
    {'joint_positions': []}, {'joint_names': []},
    {'joint_positions': [float('nan')]}, {'sim_time': -1.0},
    {'sim_time': float('inf')},
    {'joint_names': ['hip', 'hip'], 'joint_positions': [0.0, 0.0]},
])
def test_invalid_frames_are_rejected(changes):
    with pytest.raises(ValueError):
        checked_frame(frame(**changes))


class RecordingConnection:
    def __init__(self):
        self.messages = []
        self.closed = False

    def send(self, message):
        self.messages.append(message)

    def close(self):
        self.closed = True


def exercise_worker(frames, duration=1.0, controller="scripted"):
    module = ModuleType('x2_recovery.runtime')

    class TestRuntime:
        control_dt = 0.02
        closed = False

        def __init__(self, **kwargs):
            self.frames = iter(frames)

        def reset(self, seed):
            pass

        def step(self):
            value = next(self.frames)
            if isinstance(value, Exception):
                raise value
            return value

        def close(self):
            TestRuntime.closed = True

    module.RecoveryRuntime = TestRuntime
    connection = RecordingConnection()
    with patch.dict(sys.modules, {'x2_recovery.runtime': module}):
        run_episode({'controller': controller, 'checkpoint': '', 'render': False,
                     'seed': 0, 'realtime': False, 'max_sim_duration_s': duration}, connection)
    assert connection.closed and TestRuntime.closed
    return connection.messages


def test_success_comes_from_runtime_and_preserves_telemetry():
    messages = exercise_worker([frame(success=True)])
    assert messages[0] == ('frame', checked_frame(frame(success=True)))
    assert messages[-1][1]['success'] is True


def test_invalid_state_overrides_success():
    messages = exercise_worker([frame(success=True, invalid=True)])
    assert messages[-1][1]['success'] is False
    assert messages[-1][1]['reason'] == 'invalid simulator state'


@pytest.mark.parametrize('limits', [{'ok': False}, {}, None])
def test_trajectory_violation_or_missing_monitor_overrides_posture_success(limits):
    messages = exercise_worker([frame(success=True, trajectory_limits=limits)])
    assert messages[-1][1]['success'] is False
    assert 'trajectory limits' in messages[-1][1]['reason']


def test_simulation_timeout_prevents_infinite_episode():
    messages = exercise_worker([frame(), frame(sim_time=0.04)], duration=0.04)
    assert messages[-1][1] == {'success': False, 'reason': 'simulation duration timeout'}


def test_simulator_failure_is_reported_and_closed():
    messages = exercise_worker([RuntimeError('deliberate test failure')])
    assert messages[-1][1]['success'] is False
    assert 'deliberate test failure' in messages[-1][1]['reason']


def test_stalled_simulation_clock_fails():
    messages = exercise_worker([frame(), frame()])
    assert 'time did not advance' in messages[-1][1]['reason']


def test_full_recovery_audits_after_early_success():
    frames = [frame(success=True, clean_stance_success=True),
              frame(sim_time=15., success=True, clean_stance_success=False)]
    messages = exercise_worker(frames, duration=15., controller='full_recovery')
    assert len([m for m in messages if m[0] == 'frame']) == 2
    assert messages[-1][1]['success'] is False


def test_full_recovery_requires_clean_standing_at_full_horizon():
    frames = [frame(success=True, clean_stance_success=True),
              frame(sim_time=15., success=True, clean_stance_success=True)]
    messages = exercise_worker(frames, duration=15., controller='full_recovery')
    assert messages[-1][1]['success'] is True


def test_shortened_full_recovery_is_timeout_even_if_standing():
    messages = exercise_worker([frame(success=True, clean_stance_success=True)],
                              duration=.02, controller='full_recovery')
    assert messages[-1][1]['success'] is False
